"""Leakage-free feature engineering for AegisKEV.

Every time-dependent feature in this module is strictly causal: a value for a
row anchored on day ``D`` may only use information that was knowable on ``D``.
In particular, a CVE's 180-day KEV label is revealed to the prior only after
its outcome window has fully elapsed (``effective_anchor_date + 180 days``).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("aegiskev.features")

HORIZON_DAYS = 180
MATURATION_DAYS = 180
EMBEDDING_PREFIX = "text_emb_"


# --------------------------------------------------------------------------- #
# Causal group priors
# --------------------------------------------------------------------------- #
def compute_matured_causal_group_prior(
    df: pd.DataFrame,
    group_col: str,
    out_prefix: str,
    m: float,
) -> pd.DataFrame:
    """Strictly causal Bayesian m-estimate KEV prior for ``group_col``.

    For a row anchored on day ``D`` the prior is built only from labels whose
    maturation date (``effective_anchor_date + 180 days``) is ``<= D``. Rows
    sharing the same anchor day are scored as one batch, so same-day
    disclosures can never leak to each other.

    Returns a copy of ``df`` with ``{out_prefix}_kev_prior`` and
    ``{out_prefix}_hist_count`` added.
    """
    required = {group_col, "effective_anchor_date", "y_180d"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns for causal prior: {sorted(missing)}")

    out = df.copy()
    prior_col = f"{out_prefix}_kev_prior"
    count_col = f"{out_prefix}_hist_count"
    out[prior_col] = 0.0
    out[count_col] = 0

    work = out.loc[out["effective_anchor_date"].notna()].copy()
    if work.empty:
        return out

    work["_maturation_date"] = work["effective_anchor_date"] + pd.Timedelta(
        days=MATURATION_DAYS
    )
    # NaN group keys still get their own bucket instead of being dropped.
    work[group_col] = work[group_col].fillna("__missing__")

    labeled = (
        work.loc[work["y_180d"].notna(), ["_maturation_date", group_col, "y_180d"]]
        .sort_values("_maturation_date", kind="stable")
    )
    queue: deque[tuple[pd.Timestamp, object, float]] = deque(
        (row[0], row[1], float(row[2]))
        for row in labeled.itertuples(index=False, name=None)
    )

    totals: dict[object, list[float]] = {}
    total_sum = 0.0
    total_count = 0

    for day in np.sort(work["effective_anchor_date"].unique()):
        day = pd.Timestamp(day)
        # Reveal every label whose outcome window closed on or before ``day``.
        while queue and queue[0][0] <= day:
            _, group, label = queue.popleft()
            bucket = totals.setdefault(group, [0.0, 0.0])
            bucket[0] += label
            bucket[1] += 1.0
            total_sum += label
            total_count += 1

        base_rate = total_sum / total_count if total_count else 0.0
        same_day = work["effective_anchor_date"] == day
        for group, index in work.loc[same_day].groupby(group_col, dropna=False).groups.items():
            group_sum, group_count = totals.get(group, [0.0, 0.0])
            out.loc[index, prior_col] = (group_sum + m * base_rate) / (group_count + m)
            out.loc[index, count_col] = int(group_count)

    return out


# --------------------------------------------------------------------------- #
# Domain interactions
# --------------------------------------------------------------------------- #
def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    return df[column].fillna(False).astype(bool)


def add_domain_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add security-domain interactions that the raw NVD fields hide."""
    out = df.copy()
    epss = _numeric(out, "epss").clip(lower=0.0, upper=1.0)
    cvss_norm = (_numeric(out, "cvss_base_score") / 10.0).clip(lower=0.0, upper=1.0)

    if "anchor_date" in out.columns:
        out["epss_same_day_pct"] = epss.groupby(out["anchor_date"]).rank(pct=True)
    else:
        out["epss_same_day_pct"] = epss.rank(pct=True)

    out["epss_minus_cvss_norm"] = epss - cvss_norm
    out["epss_x_cvss"] = epss * cvss_norm
    out["exploit_without_patch_race"] = (
        _flag(out, "has_exploit_tag").astype(int) - _flag(out, "has_patch_tag").astype(int)
    )
    out["zero_click_network_flaw"] = (
        _flag(out, "is_network_av")
        & _flag(out, "is_low_ac")
        & _flag(out, "is_priv_none")
        & _flag(out, "is_ui_none")
    ).astype(int)

    exploitability = _numeric(out, "cvss_exploitability_score")
    impact = _numeric(out, "cvss_impact_score")
    ratio = np.full(len(out), np.nan)
    valid = exploitability.notna() & impact.notna() & (impact != 0)
    ratio[valid.to_numpy()] = (
        exploitability[valid] / impact[valid]
    ).to_numpy()
    out["exploit_to_impact_ratio"] = ratio
    return out


def add_epss_logit(df: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    """Logit of EPSS, clipped away from 0/1 to keep the transform finite."""
    out = df.copy()
    p = _numeric(out, "epss").clip(lower=epsilon, upper=1.0 - epsilon)
    out["epss_logit"] = np.log(p / (1.0 - p))
    return out


# --------------------------------------------------------------------------- #
# Dense text embeddings (fit on train only)
# --------------------------------------------------------------------------- #
def _encode_texts(
    texts: Iterable[object], model_name: str
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    cleaned = ["" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text) for text in texts]
    return np.asarray(
        model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        dtype="float32",
    )


def _attach_embedding_columns(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    width = values.shape[1]
    for index in range(width):
        out[f"{EMBEDDING_PREFIX}{index:02d}"] = values[:, index]
    return out


def add_dense_text_embeddings(
    train_df: pd.DataFrame,
    other_dfs: Sequence[pd.DataFrame],
    model_name: str = "BAAI/bge-small-en-v1.5",
    n_components: int = 24,
    text_col: str = "description_en",
) -> list[pd.DataFrame]:
    """BGE embeddings reduced with PCA fit **strictly on ``train_df``**.

    Returns ``[train_out, *other_outs]`` in the input order. Importing
    ``sentence-transformers`` / ``scikit-learn`` is deferred so the rest of the
    module stays importable on a minimal environment.
    """
    from sklearn.decomposition import PCA

    if text_col not in train_df.columns:
        raise KeyError(f"missing text column: {text_col}")

    train_embedding = _encode_texts(train_df[text_col], model_name)
    width = min(n_components, train_embedding.shape[0], train_embedding.shape[1])
    if width < n_components:
        LOGGER.warning(
            "Capping text PCA components at %s (requested %s)", width, n_components
        )
    pca = PCA(n_components=width, random_state=42)
    train_reduced = pca.fit_transform(train_embedding)

    outputs = [_attach_embedding_columns(train_df, train_reduced)]
    for frame in other_dfs:
        if text_col not in frame.columns:
            raise KeyError(f"missing text column: {text_col}")
        reduced = pca.transform(_encode_texts(frame[text_col], model_name))
        outputs.append(_attach_embedding_columns(frame, reduced))
    return outputs


def build_tfidf_svd_pipeline(
    n_components: int = 24,
    random_state: int = 42,
    min_df: int = 2,
):
    """Backup TF-IDF + SVD text pipeline (fit on train, saved for ablations)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ]
    )
