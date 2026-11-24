"""Metric, conformal and end-to-end training tests for Part 3."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.training.train_pipeline import (
    ALPHA,
    STAGE_NAMES,
    conformal_set_metrics,
    expected_calibration_error,
    paired_bootstrap_delta_pr_auc,
    risk_weighted_capture,
    run_training,
    split_conformal_quantile,
    top_k_metrics,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_ece_matches_hand_computation():
    ece = expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=10)
    assert ece == pytest.approx(0.15)


def test_rvc_weights_captured_positives_by_cvss():
    y = [1, 1, 0, 0]
    score = [0.9, 0.8, 0.7, 0.6]
    risk = [9.0, 1.0, 5.0, 5.0]

    assert risk_weighted_capture(y, score, risk, k=1) == pytest.approx(0.9)
    assert risk_weighted_capture(y, score, risk, k=2) == pytest.approx(1.0)
    assert risk_weighted_capture(y, score, risk, k=10) == pytest.approx(1.0)


def test_rvc_handles_no_positives():
    assert risk_weighted_capture([0, 0], [0.9, 0.8], [5.0, 6.0], k=1) == 0.0


def test_top_one_percent_confusion_matrix():
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    metrics = top_k_metrics(y, score, fraction=0.01)

    assert metrics["k"] == 1
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1
    assert metrics["tn"] == 8
    assert metrics["precision_top_1pct"] == pytest.approx(1.0)
    assert metrics["recall_top_1pct"] == pytest.approx(0.5)
    assert metrics["f1_top_1pct"] == pytest.approx(2 / 3)


def test_paired_bootstrap_detects_improvement():
    y = np.array([0, 0, 1, 1] * 25)
    good = np.clip(y * 0.6 + np.linspace(0.1, 0.3, len(y)), 0.0, 1.0)
    bad = np.full(len(y), 0.5)

    result = paired_bootstrap_delta_pr_auc(y, good, bad, n_boot=200, seed=1)

    assert result["delta_mean"] > 0.1
    assert result["ci_low"] > 0.0
    assert result["ci_high"] <= 1.0
    assert result["p_value_one_sided"] < 0.05


def test_paired_bootstrap_identical_scores_is_null():
    y = np.array([0, 1, 0, 1] * 10)
    scores = np.linspace(0.1, 0.9, len(y))

    result = paired_bootstrap_delta_pr_auc(y, scores, scores, n_boot=50, seed=2)

    assert result["delta_mean"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_value_one_sided"] == 1.0


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #
def test_split_conformal_quantile_finite_sample_bound():
    q_hat = split_conformal_quantile(
        [1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4], alpha=ALPHA
    )
    # scores = [0.1, 0.2, 0.4, 0.4]; ceil(5*0.9)/4 > 1 -> clipped to max score.
    assert q_hat == pytest.approx(0.4)


def test_split_conformal_quantile_shrinks_with_larger_alpha():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(0.0, 1.0, 200)

    assert split_conformal_quantile(y, p, alpha=0.05) >= split_conformal_quantile(
        y, p, alpha=0.20
    )


def test_conformal_sets_flag_ambiguity_and_emptiness():
    metrics = conformal_set_metrics([1, 0], [0.5, 0.5], q_hat=0.6)

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["ambiguous_fraction"] == pytest.approx(1.0)
    assert metrics["avg_set_size"] == pytest.approx(2.0)

    empty = conformal_set_metrics([1, 0], [0.05, 0.95], q_hat=0.01)
    assert empty["empty_fraction"] == pytest.approx(1.0)
    assert empty["coverage"] == 0.0


# --------------------------------------------------------------------------- #
# End-to-end synthetic smoke
# --------------------------------------------------------------------------- #
def _synthetic_split(n: int, start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    span = (pd.Timestamp(end) - pd.Timestamp(start)).days
    dates = pd.to_datetime(start) + pd.to_timedelta(
        rng.integers(0, span + 1, n), unit="D"
    )

    epss = rng.beta(2.0, 5.0, n)
    cvss = rng.uniform(2.0, 10.0, n)
    logit = 4.0 * epss + 0.35 * (cvss - 5.0) - 3.2
    probability = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, probability)
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == n:
        y[-1] = 0

    missing = rng.random(n) < 0.1
    epss_missing = epss.copy()
    epss_missing[missing] = np.nan
    clipped = np.clip(epss_missing, 1e-6, 1.0 - 1e-6)

    return pd.DataFrame(
        {
            "effective_anchor_date": dates,
            "y_180d": y,
            "cvss_base_score": cvss,
            "cvss_exploitability_score": cvss * 0.4,
            "cvss_impact_score": cvss * 0.6,
            "is_network_av": rng.integers(0, 2, n),
            "is_low_ac": rng.integers(0, 2, n),
            "is_priv_none": rng.integers(0, 2, n),
            "is_ui_none": rng.integers(0, 2, n),
            "scope_changed": rng.integers(0, 2, n),
            "cwe_count": rng.integers(1, 4, n),
            "cpe_config_count": rng.integers(0, 4, n),
            "cpe_vendor_count": rng.integers(0, 4, n),
            "reference_count": rng.integers(0, 12, n),
            "has_vendor_advisory_ref": rng.integers(0, 2, n),
            "has_patch_tag": rng.integers(0, 2, n),
            "has_exploit_tag": rng.integers(0, 2, n),
            "epss": epss_missing,
            "epss_percentile": pd.Series(epss).rank(pct=True).to_numpy(),
            "epss_logit": np.log(clipped / (1.0 - clipped)),
            "epss_missing_flag": missing.astype(int),
            "cwe_kev_prior": rng.uniform(0.0, 0.5, n),
            "primary_vendor_kev_prior": rng.uniform(0.0, 0.5, n),
            "epss_same_day_pct": rng.uniform(0.0, 1.0, n),
            "epss_minus_cvss_norm": epss - cvss / 10.0,
            "exploit_without_patch_race": rng.integers(-1, 2, n),
            "zero_click_network_flaw": rng.integers(0, 2, n),
            "exploit_to_impact_ratio": rng.uniform(0.1, 1.0, n),
            "description_en": [
                f"synthetic description {index} remote code execution"
                for index in range(n)
            ],
            **{
                f"text_emb_{index:02d}": rng.normal(size=n)
                for index in range(4)
            },
        }
    )


def test_train_pipeline_end_to_end(tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    frames = {
        "train": _synthetic_split(240, "2023-06-01", "2024-06-30", 11),
        "val_hpo": _synthetic_split(70, "2024-07-01", "2024-11-30", 12),
        "val_cal": _synthetic_split(70, "2024-12-01", "2025-03-16", 13),
        "test": _synthetic_split(90, "2025-03-17", "2025-07-31", 14),
    }
    for name, frame in frames.items():
        frame.to_parquet(curated / f"{name}.parquet", index=False)

    artifact_dir = tmp_path / "champion"
    summary = run_training(
        curated_dir=curated,
        artifact_dir=artifact_dir,
        trials=2,
        n_boot=50,
        skip_mlflow_ui=True,
        mlflow_dir=tmp_path / "mlruns",
        seed=7,
    )

    assert summary["champion"]["name"] in {"lightgbm", "xgboost", "catboost"}
    assert set(summary["models"]) == {
        "lightgbm",
        "xgboost",
        "catboost",
        "stacked_ensemble",
        "Rule_CVSS_Only",
        "Rule_EPSS_Only",
        "LogisticRegression",
        "RandomForest",
    }
    assert all(0.0 <= m["pr_auc"] <= 1.0 for m in summary["models"].values())
    assert 0.0 <= summary["conformal"]["q_hat"] <= 1.0
    assert 0.0 <= summary["conformal"]["coverage"] <= 1.0
    assert 0.0 <= summary["conformal"]["ambiguous_fraction"] <= 1.0
    assert set(summary["ablation"]) == set(STAGE_NAMES)
    assert summary["ablation"][STAGE_NAMES[-1]]["n_features"] >= summary["ablation"][
        STAGE_NAMES[0]
    ]["n_features"]
    assert summary["shap_top_features"], "expected a non-empty SHAP ranking"

    summary_path = artifact_dir / "metrics_summary.json"
    bundle_path = artifact_dir / "champion_bundle.joblib"
    assert summary_path.exists() and bundle_path.exists()

    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk["experiment"] == "AegisKEV_180d_Exploit_Prediction"

    bundle = joblib.load(bundle_path)
    assert bundle["conformal_alpha"] == ALPHA
    assert hasattr(bundle["model"], "predict_proba")
    probabilities = bundle["model"].predict_proba(
        frames["test"][bundle["feature_columns"]].astype("float64")
    )[:, 1]
    assert len(probabilities) == len(frames["test"])
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()

    experiment_dir = tmp_path / "mlruns"
    assert experiment_dir.exists()
    assert any(path.is_file() for path in experiment_dir.rglob("*"))
