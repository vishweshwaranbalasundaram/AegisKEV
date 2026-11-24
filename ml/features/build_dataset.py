"""Build the leakage-free AegisKEV modelling dataset.

Pipeline
--------
1. Join NVD features with point-in-time EPSS anchors and KEV dates.
2. Drop CVEs already in KEV at triage time, then label
   ``y_180d = 1`` iff ``anchor < date_added <= anchor + 180d``.
3. Right-censor: anchors newer than ``max(date_added) - 180d`` have not had a
   full observation window, so they go to ``unmatured_recent.parquet``.
4. Add causal group priors, domain interactions, EPSS logit and text features.
5. Split chronologically and write parquet splits + a hashed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ml.features.advanced_features import (
    add_dense_text_embeddings,
    add_domain_interaction_features,
    add_epss_logit,
    build_tfidf_svd_pipeline,
    compute_matured_causal_group_prior,
)

LOGGER = logging.getLogger("aegiskev.dataset")

HORIZON_DAYS = 180
EPSS_START = pd.Timestamp("2021-04-14")

SPLIT_ORDER = ("train", "val_hpo", "val_cal", "test")

# Fixed chronological windows (inclusive).
BOUNDARIES = {
    "train_end": pd.Timestamp("2024-06-30"),
    "val_hpo_start": pd.Timestamp("2024-07-01"),
    "val_hpo_end": pd.Timestamp("2024-11-30"),
    "val_cal_start": pd.Timestamp("2024-12-01"),
    "val_cal_end": pd.Timestamp("2025-03-16"),
    "test_start": pd.Timestamp("2025-03-17"),
}

FALLBACK_QUANTILES = (0.60, 0.75, 0.85)

PRIOR_SPECS = (
    ("cwe_primary", "cwe", 20.0),
    ("primary_vendor", "vendor", 15.0),
)

# Schema contract with ml/ingestion/sync_data.py. Every name below must exist
# in nvd_parsed.parquet / epss_anchored.parquet; prepare_dataset fails loudly
# instead of silently producing a feature frame with missing columns.
REQUIRED_NVD_COLUMNS: tuple[str, ...] = (
    "cve_id",
    "published",
    "description_en",
    "cwe_primary",
    "primary_vendor",
    "cvss_base_score",
    "cvss_exploitability_score",
    "cvss_impact_score",
    "is_network_av",
    "is_low_ac",
    "is_priv_none",
    "is_ui_none",
    "scope_changed",
    "cpe_config_count",
    "cpe_vendor_count",
    "reference_count",
    "has_vendor_advisory_ref",
    "has_patch_tag",
    "has_exploit_tag",
)

REQUIRED_EPSS_COLUMNS: tuple[str, ...] = ("cve", "epss", "percentile", "anchor_date")

TFIDF_PREFIX = "tfidf_svd_"


# --------------------------------------------------------------------------- #
# Loading / joining
# --------------------------------------------------------------------------- #
def load_inputs(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nvd = pd.read_parquet(Path(raw_dir) / "nvd" / "nvd_parsed.parquet")
    epss = pd.read_parquet(Path(raw_dir) / "epss_anchored.parquet")
    kev = pd.read_parquet(Path(raw_dir) / "kev" / "kev_catalog.parquet")
    return nvd, epss, kev


def exclude_already_kev(df: pd.DataFrame) -> pd.DataFrame:
    """Drop CVEs that were already in KEV on their triage day."""
    already = df["date_added"].notna() & (
        df["date_added"] <= df["effective_anchor_date"]
    )
    return df.loc[~already].copy()


def assign_target(df: pd.DataFrame) -> pd.DataFrame:
    """``y_180d = 1`` iff KEV entry falls inside ``(anchor, anchor + 180d]``."""
    out = df.copy()
    horizon = out["effective_anchor_date"] + pd.Timedelta(days=HORIZON_DAYS)
    known = out["date_added"].notna()
    in_window = (
        known
        & (out["date_added"] > out["effective_anchor_date"])
        & (out["date_added"] <= horizon)
    )
    out["y_180d"] = in_window.astype(int)
    return out


def prepare_dataset(
    nvd: pd.DataFrame,
    epss: pd.DataFrame,
    kev: pd.DataFrame,
    start_date: pd.Timestamp = EPSS_START,
) -> pd.DataFrame:
    nvd = nvd.copy()
    # Accept parquet files written by older ingestion builds too.
    if "description_en" not in nvd.columns and "description" in nvd.columns:
        nvd = nvd.rename(columns={"description": "description_en"})
    if "cwe_primary" not in nvd.columns and "primary_cwe" in nvd.columns:
        nvd = nvd.rename(columns={"primary_cwe": "cwe_primary"})
    missing_nvd = [column for column in REQUIRED_NVD_COLUMNS if column not in nvd.columns]
    if missing_nvd:
        raise KeyError(
            "nvd_parsed.parquet is missing required columns: "
            f"{missing_nvd}; re-run ml.ingestion.sync_data"
        )
    nvd = nvd.rename(columns={"published": "published_date"})
    nvd["published_date"] = pd.to_datetime(nvd["published_date"], errors="coerce")
    nvd = nvd.loc[nvd["published_date"].notna() & (nvd["published_date"] >= start_date)]

    epss = epss.copy()
    missing_epss = [
        column for column in REQUIRED_EPSS_COLUMNS if column not in epss.columns
    ]
    if missing_epss:
        raise KeyError(
            "epss_anchored.parquet is missing required columns: "
            f"{missing_epss}; re-run ml.ingestion.sync_data"
        )
    epss = epss.rename(
        columns={"cve": "cve_id", "percentile": "epss_percentile"}
    )
    epss["anchor_date"] = pd.to_datetime(epss["anchor_date"], errors="coerce")
    epss = epss.dropna(subset=["anchor_date"]).drop_duplicates("cve_id", keep="first")
    epss_columns = [
        column
        for column in ("cve_id", "epss", "epss_percentile", "anchor_date", "model_version")
        if column in epss.columns
    ]

    frame = nvd.merge(epss[epss_columns], on="cve_id", how="left")
    # Missing EPSS is kept as NaN (never imputed at ingest); the flag lets the
    # model learn from its absence instead of silently dropping negatives.
    frame["epss_missing_flag"] = (
        frame["epss"].isna().astype(int) if "epss" in frame.columns else 1
    )
    # A CVE with no archived EPSS day still gets a valid published+1 anchor.
    frame["effective_anchor_date"] = frame["anchor_date"].fillna(
        frame["published_date"] + pd.Timedelta(days=1)
    )

    kev_dates = kev[["cve_id", "date_added"]].copy()
    kev_dates["date_added"] = pd.to_datetime(kev_dates["date_added"], errors="coerce")
    frame = frame.merge(kev_dates, on="cve_id", how="left")

    frame = frame.loc[frame["effective_anchor_date"].notna()]
    frame = exclude_already_kev(frame)
    return assign_target(frame)


# --------------------------------------------------------------------------- #
# Right-censoring
# --------------------------------------------------------------------------- #
def apply_censoring(
    df: pd.DataFrame, kev_cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split matured rows from rows whose 180-day window is not observed yet."""
    cutoff = pd.Timestamp(kev_cutoff) - pd.Timedelta(days=HORIZON_DAYS)
    matured = df["effective_anchor_date"] <= cutoff
    return df.loc[matured].copy(), df.loc[~matured].copy(), cutoff


# --------------------------------------------------------------------------- #
# Chronological splitting
# --------------------------------------------------------------------------- #
def _fixed_masks(dates: pd.Series, boundaries: dict[str, pd.Timestamp]) -> dict[str, pd.Series]:
    return {
        "train": dates <= boundaries["train_end"],
        "val_hpo": (dates >= boundaries["val_hpo_start"])
        & (dates <= boundaries["val_hpo_end"]),
        "val_cal": (dates >= boundaries["val_cal_start"])
        & (dates <= boundaries["val_cal_end"]),
        "test": dates >= boundaries["test_start"],
    }


def _quantile_splits(
    df: pd.DataFrame, quantiles: Sequence[float]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Timestamp]]:
    unique_days = np.sort(df["effective_anchor_date"].dropna().unique())
    total = len(unique_days)
    if total < 8:
        raise ValueError(
            f"need at least 8 distinct anchor days for a quantile fallback, got {total}"
        )

    cut1 = max(1, min(total - 3, int(np.floor(total * quantiles[0]))))
    cut2 = max(cut1 + 1, min(total - 2, int(np.floor(total * quantiles[1]))))
    cut3 = max(cut2 + 1, min(total - 1, int(np.floor(total * quantiles[2]))))

    day_sets = {
        "train": unique_days[:cut1],
        "val_hpo": unique_days[cut1:cut2],
        "val_cal": unique_days[cut2:cut3],
        "test": unique_days[cut3:],
    }
    splits = {
        name: df.loc[df["effective_anchor_date"].isin(days)].copy()
        for name, days in day_sets.items()
    }
    edges = {
        name: pd.Timestamp(days.min()) for name, days in day_sets.items() if len(days)
    }
    return splits, edges


def chronological_split(
    df: pd.DataFrame,
    boundaries: dict[str, pd.Timestamp] | None = None,
    quantiles: Sequence[float] = FALLBACK_QUANTILES,
    min_rows: int = 1,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Return ``(splits, scheme)`` using fixed windows or a quantile fallback.

    Fixed windows are used only when every split has at least ``min_rows``
    rows; otherwise the dates are cut by quantile so reduced year subsets still
    produce four contiguous, non-empty blocks.
    """
    if df["effective_anchor_date"].isna().any():
        raise ValueError("cannot split rows with missing effective_anchor_date")
    boundaries = boundaries or BOUNDARIES
    dates = df["effective_anchor_date"]

    masks = _fixed_masks(dates, boundaries)
    fixed = {name: df.loc[mask].copy() for name, mask in masks.items()}
    if all(len(frame) >= min_rows for frame in fixed.values()):
        return fixed, "fixed"

    LOGGER.warning(
        "Fixed split windows are empty/small (rows=%s); using quantile fallback",
        {name: len(frame) for name, frame in fixed.items()},
    )
    splits, _ = _quantile_splits(df, quantiles)
    return splits, "quantile"


# --------------------------------------------------------------------------- #
# Guards / manifest
# --------------------------------------------------------------------------- #
def assert_no_cisa_columns(df: pd.DataFrame) -> None:
    leaks = [column for column in df.columns if str(column).lower().startswith("cisa")]
    assert not leaks, f"leaky CISA columns present: {leaks}"


def assert_split_monotonic(splits: dict[str, pd.DataFrame]) -> None:
    assert all(len(splits[name]) > 0 for name in SPLIT_ORDER), {
        name: len(splits[name]) for name in SPLIT_ORDER
    }
    for left, right in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        left_max = splits[left]["effective_anchor_date"].max()
        right_min = splits[right]["effective_anchor_date"].min()
        assert left_max < right_min, (
            f"non-monotonic split boundary: {left} max {left_max} >= "
            f"{right} min {right_min}"
        )


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop label-timing columns and enforce the CISA leakage guard."""
    drop = [
        column
        for column in df.columns
        if column == "date_added" or str(column).lower().startswith("cisa")
    ]
    out = df.drop(columns=drop, errors="ignore")
    assert_no_cisa_columns(out)
    assert "date_added" not in out.columns, "date_added must not survive into features"
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    split_paths: dict[str, Path],
    splits: dict[str, pd.DataFrame],
    kev_cutoff: pd.Timestamp,
    censoring_cutoff: pd.Timestamp,
    scheme: str,
    unmatured_rows: int,
    text_features: dict[str, bool],
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kev_cutoff": pd.Timestamp(kev_cutoff).isoformat(),
        "censoring_cutoff": pd.Timestamp(censoring_cutoff).isoformat(),
        "horizon_days": HORIZON_DAYS,
        "split_scheme": scheme,
        "row_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "positive_prevalence": {
            name: round(float(frame["y_180d"].mean()), 6)
            for name, frame in splits.items()
        },
        "date_ranges": {
            name: {
                "min": pd.Timestamp(frame["effective_anchor_date"].min()).isoformat(),
                "max": pd.Timestamp(frame["effective_anchor_date"].max()).isoformat(),
            }
            for name, frame in splits.items()
        },
        "feature_columns": int(splits["train"].shape[1]),
        "unmatured_rows": int(unmatured_rows),
        "text_features": text_features,
        "sha256": {
            name: sha256_file(path) for name, path in split_paths.items()
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _attach_tfidf_columns(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for index in range(values.shape[1]):
        out[f"{TFIDF_PREFIX}{index:02d}"] = values[:, index]
    return out


def run_pipeline(
    raw_dir: Path = Path("data/raw"),
    curated_dir: Path = Path("data/curated"),
    artifacts_dir: Path = Path("artifacts"),
    skip_embeddings: bool = False,
    skip_tfidf: bool = False,
    boundaries: dict[str, pd.Timestamp] | None = None,
) -> dict:
    raw_dir, curated_dir, artifacts_dir = Path(raw_dir), Path(curated_dir), Path(artifacts_dir)
    curated_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "preprocessors").mkdir(parents=True, exist_ok=True)

    nvd, epss, kev = load_inputs(raw_dir)
    frame = prepare_dataset(nvd, epss, kev)

    kev_cutoff = pd.to_datetime(kev["date_added"], errors="coerce").max()
    if pd.isna(kev_cutoff):
        raise ValueError("KEV catalogue has no usable date_added values")
    matured, unmatured, censoring_cutoff = apply_censoring(frame, kev_cutoff)
    unmatured_path = curated_dir / "unmatured_recent.parquet"
    unmatured.to_parquet(unmatured_path, index=False)
    LOGGER.info(
        "Matured rows: %s | right-censored rows: %s (cutoff %s)",
        len(matured),
        len(unmatured),
        censoring_cutoff.date(),
    )

    for group_col, prefix, m in PRIOR_SPECS:
        matured = compute_matured_causal_group_prior(matured, group_col, prefix, m)
    matured = add_domain_interaction_features(matured)
    matured = add_epss_logit(matured)

    splits, scheme = chronological_split(matured, boundaries)
    LOGGER.info("Split scheme: %s | rows: %s", scheme, {k: len(v) for k, v in splits.items()})

    text_features = {"dense_embeddings": False, "tfidf_svd": False}

    if not skip_embeddings:
        train, *others = add_dense_text_embeddings(
            splits["train"],
            [splits["val_hpo"], splits["val_cal"], splits["test"]],
        )
        splits = dict(zip(SPLIT_ORDER, [train, *others]))
        text_features["dense_embeddings"] = True

    if not skip_tfidf:
        import joblib

        pipeline = build_tfidf_svd_pipeline()
        train_text = splits["train"]["description_en"].fillna("")
        pipeline.fit(train_text)
        for name in SPLIT_ORDER:
            values = pipeline.transform(splits[name]["description_en"].fillna(""))
            splits[name] = _attach_tfidf_columns(splits[name], values)
        joblib.dump(pipeline, artifacts_dir / "preprocessors" / "text_pipeline.joblib")
        text_features["tfidf_svd"] = True

    splits = {name: finalize_features(frame) for name, frame in splits.items()}
    assert_split_monotonic(splits)

    split_paths = {name: curated_dir / f"{name}.parquet" for name in SPLIT_ORDER}
    for name, path in split_paths.items():
        splits[name].to_parquet(path, index=False)

    manifest = build_manifest(
        split_paths,
        splits,
        kev_cutoff,
        censoring_cutoff,
        scheme,
        len(unmatured),
        text_features,
    )
    manifest_path = curated_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Wrote splits + %s", manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AegisKEV modelling dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="skip BGE + PCA dense text embeddings",
    )
    parser.add_argument(
        "--skip-tfidf", action="store_true", help="skip the TF-IDF + SVD backup pipeline"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    run_pipeline(
        raw_dir=args.raw_dir,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        skip_embeddings=args.skip_embeddings,
        skip_tfidf=args.skip_tfidf,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
