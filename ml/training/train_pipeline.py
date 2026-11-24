"""Train, calibrate, explain and benchmark the AegisKEV 180-day model suite.

Design contract
---------------
* Test split is touched once for final scoring only.
* Learned models are calibrated on ``val_cal`` with
  ``CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")``.
* Conformal prediction uses split nonconformity at ``alpha=0.10``.
* Champion selection uses ``val_hpo`` PR-AUC, never the test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, roc_auc_score

LOGGER = logging.getLogger("aegiskev.training")

EXPERIMENT_NAME = "AegisKEV_180d_Exploit_Prediction"
SPLITS = ("train", "val_hpo", "val_cal", "test")
MODEL_KINDS = ("lightgbm", "xgboost", "catboost")
TOP_FRACTION = 0.01
ALPHA = 0.10

BASE_FEATURES = (
    "cvss_base_score",
    "cvss_exploitability_score",
    "cvss_impact_score",
    "is_network_av",
    "is_low_ac",
    "is_priv_none",
    "is_ui_none",
    "scope_changed",
    "cwe_count",
    "cpe_config_count",
    "cpe_vendor_count",
    "reference_count",
    "has_vendor_advisory_ref",
    "has_patch_tag",
    "has_exploit_tag",
    "epss",
    "epss_percentile",
    "epss_logit",
    "epss_missing_flag",
)

STAGE2_FEATURES = (
    "cwe_kev_prior",
    "primary_vendor_kev_prior",
    "epss_same_day_pct",
    "epss_minus_cvss_norm",
    "exploit_without_patch_race",
    "zero_click_network_flaw",
    "exploit_to_impact_ratio",
)

TEXT_PREFIXES = ("text_emb_", "emb_pca_", "desc_svd_", "tfidf_svd_")

STAGE_NAMES = (
    "Stage_1_Base_NVD_EPSS",
    "Stage_2_Plus_Causal_Priors_Race",
    "Stage_3_Plus_Text_Embeddings",
)


# --------------------------------------------------------------------------- #
# Metrics (pure, unit-tested)
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10
) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.clip(np.asarray(list(y_prob), dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        mask = bins == index
        if not mask.any():
            continue
        error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def risk_weighted_capture(
    y_true: Iterable[int],
    y_score: Iterable[float],
    risk: Iterable[float],
    k: int,
) -> float:
    """Share of positive-CVE risk captured in the top-``k`` ranked rows."""
    y = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    weights = np.nan_to_num(np.asarray(list(risk), dtype=float), nan=0.0)
    if len(y) == 0:
        return 0.0
    k = min(max(int(k), 1), len(y))
    top = np.argsort(-score, kind="stable")[:k]
    total = float((weights * y).sum())
    if total <= 0.0:
        return 0.0
    return float((weights[top] * y[top]).sum() / total)


def top_k_metrics(
    y_true: Iterable[int], y_score: Iterable[float], fraction: float = TOP_FRACTION
) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)
    n = len(y)
    if n == 0:
        return {
            "k": 0,
            "recall_top_1pct": 0.0,
            "precision_top_1pct": 0.0,
            "f1_top_1pct": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    k = max(1, int(math.ceil(n * fraction)))
    top = np.argsort(-score, kind="stable")[:k]
    predicted = np.zeros(n, dtype=int)
    predicted[top] = 1
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = k - tp
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = n - k - fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / k if k else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "k": k,
        "recall_top_1pct": float(recall),
        "precision_top_1pct": float(precision),
        "f1_top_1pct": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def paired_bootstrap_delta_pr_auc(
    y_true: Iterable[int],
    score_model: Iterable[float],
    score_reference: Iterable[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap CI + one-sided p-value for ΔPR-AUC (model - reference)."""
    y = np.asarray(list(y_true), dtype=int)
    a = np.asarray(list(score_model), dtype=float)
    b = np.asarray(list(score_reference), dtype=float)
    n = len(y)
    if n == 0:
        return {
            "delta_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value_one_sided": 1.0,
            "n_boot": 0,
        }
    rng = np.random.default_rng(seed)
    deltas = np.zeros(max(int(n_boot), 1), dtype=float)
    for index in range(len(deltas)):
        sample = rng.integers(0, n, n)
        y_sample = y[sample]
        if y_sample.min() == y_sample.max():
            continue
        deltas[index] = average_precision_score(
            y_sample, a[sample]
        ) - average_precision_score(y_sample, b[sample])
    low, high = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = (float((deltas <= 0).sum()) + 1.0) / (len(deltas) + 1.0)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value_one_sided": float(p_value),
        "n_boot": int(len(deltas)),
    }


# --------------------------------------------------------------------------- #
# Split conformal
# --------------------------------------------------------------------------- #
def split_conformal_quantile(
    y_cal: Iterable[int], p_cal: Iterable[float], alpha: float = ALPHA
) -> float:
    """Finite-sample split-conformal quantile of ``1 - p(true label)``."""
    y = np.asarray(list(y_cal), dtype=int)
    p = np.clip(np.asarray(list(p_cal), dtype=float), 0.0, 1.0)
    if len(y) == 0:
        raise ValueError("cannot compute a conformal quantile on an empty set")
    scores = np.where(y == 1, 1.0 - p, p)
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def conformal_set_metrics(
    y_true: Iterable[int], y_prob: Iterable[float], q_hat: float
) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(y_prob), dtype=float), 0.0, 1.0)
    include_one = (1.0 - p) <= q_hat
    include_zero = p <= q_hat
    sizes = include_one.astype(int) + include_zero.astype(int)
    covered = np.where(y == 1, include_one, include_zero)
    ambiguous = include_zero & include_one
    return {
        "coverage": float(covered.mean()) if len(y) else 0.0,
        "ambiguous_fraction": float(ambiguous.mean()) if len(y) else 0.0,
        "empty_fraction": float((sizes == 0).mean()) if len(y) else 0.0,
        "avg_set_size": float(sizes.mean()) if len(y) else 0.0,
    }


def evaluate_predictions(
    y_true: Iterable[int], y_prob: Iterable[float], risk: Iterable[float] | None = None
) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.clip(np.asarray(list(y_prob), dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return {}
    single_class = len(np.unique(y)) < 2
    metrics: dict[str, float] = {
        "pr_auc": 0.0 if single_class else float(average_precision_score(y, p)),
        "roc_auc": 0.5 if single_class else float(roc_auc_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece_10": expected_calibration_error(y, p, n_bins=10),
        "positive_rate": float(y.mean()),
        "n_rows": int(len(y)),
    }
    metrics.update(top_k_metrics(y, p))
    if risk is not None:
        weights = np.asarray(list(risk), dtype=float)
        metrics["rvc_100"] = risk_weighted_capture(y, p, weights, 100)
        metrics["rvc_500"] = risk_weighted_capture(y, p, weights, 500)
    return metrics


# --------------------------------------------------------------------------- #
# Data + feature matrices
# --------------------------------------------------------------------------- #
def load_splits(curated_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in SPLITS:
        path = Path(curated_dir) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing split: {path}")
        frames[name] = pd.read_parquet(path)
    return frames


def feature_stages(df: pd.DataFrame) -> dict[str, list[str]]:
    base = [column for column in BASE_FEATURES if column in df.columns]
    stage2 = base + [column for column in STAGE2_FEATURES if column in df.columns]
    text = [column for column in df.columns if str(column).startswith(TEXT_PREFIXES)]
    stages = {
        STAGE_NAMES[0]: base,
        STAGE_NAMES[1]: stage2,
        STAGE_NAMES[2]: stage2 + text,
    }
    return {name: list(dict.fromkeys(columns)) for name, columns in stages.items()}


def build_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    matrix = df.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce")
    return matrix.astype("float64")


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["y_180d"].astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# Models + HPO
# --------------------------------------------------------------------------- #
def make_sampler(seed: int = 42):
    try:
        from optuna.samplers import GPSampler

        return GPSampler(seed=seed)
    except Exception as exc:  # torch/scipy missing or GPSampler unavailable
        LOGGER.warning("GPSampler unavailable (%s); falling back to TPESampler", exc)
        from optuna.samplers import TPESampler

        return TPESampler(seed=seed)


def suggest_params(trial, kind: str) -> dict[str, Any]:
    if kind == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if kind == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-2, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if kind == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 400, step=50),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        }
    raise ValueError(f"unknown model kind: {kind}")


def build_model(kind: str, params: dict[str, Any], y_train: np.ndarray | None = None):
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            **params,
            subsample_freq=1,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        scale_pos_weight = 1.0
        if y_train is not None and len(y_train) and 0 < y_train.sum() < len(y_train):
            scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
        return XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise ValueError(f"unknown model kind: {kind}")


def run_hpo(
    kind: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int,
    seed: int = 42,
):
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        params = suggest_params(trial, kind)
        model = build_model(kind, params, y_train)
        model.fit(X_train, y_train)
        if len(np.unique(y_val)) < 2:
            return 0.0
        probabilities = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, probabilities))

    study = optuna.create_study(direction="maximize", sampler=make_sampler(seed))
    try:
        study.optimize(objective, n_trials=max(1, n_trials), show_progress_bar=False)
    except Exception as exc:  # GPSampler can be brittle on tiny data
        LOGGER.warning(
            "HPO (%s) failed with %s (%s); retrying with TPESampler",
            kind,
            type(study.sampler).__name__,
            exc,
        )
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=max(1, n_trials), show_progress_bar=False)
    return study


def calibrate_model(model, X_cal: pd.DataFrame, y_cal: np.ndarray):
    """Platt scaling on val_cal with a frozen fitted estimator (no cv='prefit')."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Stacked ensemble (3 expanding-window folds)
# --------------------------------------------------------------------------- #
class StackedEnsemble(ClassifierMixin, BaseEstimator):
    """Chronological OOF stack of LGBM/XGB/CatBoost + EPSS + normalized CVSS."""

    def __init__(self, feature_columns: Sequence[str], params_by_kind: dict[str, dict]):
        self.feature_columns = list(feature_columns)
        self.params_by_kind = params_by_kind

    @staticmethod
    def _meta_matrix(
        base_probabilities: np.ndarray, X: pd.DataFrame
    ) -> np.ndarray:
        if "epss" in X.columns:
            epss = pd.to_numeric(X["epss"], errors="coerce").fillna(0.0).to_numpy()
        else:
            epss = np.zeros(len(X))
        if "cvss_base_score" in X.columns:
            cvss = (
                pd.to_numeric(X["cvss_base_score"], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 10.0)
                .to_numpy()
                / 10.0
            )
        else:
            cvss = np.zeros(len(X))
        return np.column_stack([base_probabilities, epss, cvss])

    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: pd.Series):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        order = np.argsort(pd.to_datetime(dates).to_numpy(), kind="stable")
        X_sorted = X.iloc[order].reset_index(drop=True)
        y_sorted = np.asarray(y)[order]
        days = pd.to_datetime(pd.Series(dates).iloc[order]).reset_index(drop=True)

        cuts = [days.quantile(q) for q in (0.25, 0.50, 0.75)]
        oof = np.full((len(X_sorted), len(MODEL_KINDS)), np.nan)

        for fold in range(1, 4):
            upper = cuts[fold] if fold < 3 else days.max()
            train_mask = days <= cuts[fold - 1]
            val_mask = (days > cuts[fold - 1]) & (days <= upper)
            if not train_mask.any() or not val_mask.any():
                continue
            for column, kind in enumerate(MODEL_KINDS):
                model = build_model(kind, self.params_by_kind[kind], y_sorted[train_mask])
                model.fit(X_sorted.loc[train_mask], y_sorted[train_mask])
                oof[val_mask.to_numpy(), column] = model.predict_proba(
                    X_sorted.loc[val_mask]
                )[:, 1]

        valid = ~np.isnan(oof).any(axis=1)
        if valid.sum() < 2:
            raise ValueError("not enough OOF rows to fit the stacking meta-model")
        meta = self._meta_matrix(oof[valid], X_sorted.loc[valid])
        self.meta_model_ = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        self.meta_model_.fit(meta, y_sorted[valid])

        self.base_models_ = {}
        for kind in MODEL_KINDS:
            model = build_model(kind, self.params_by_kind[kind], y_sorted)
            model.fit(X_sorted, y_sorted)
            self.base_models_[kind] = model
        self.classes_ = np.unique(y_sorted)
        self.n_features_in_ = X_sorted.shape[1]
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "meta_model_"):
            raise RuntimeError("StackedEnsemble must be fitted before prediction")
        base = np.column_stack(
            [_positive_proba(self.base_models_[kind], X) for kind in MODEL_KINDS]
        )
        meta = self._meta_matrix(base, X)
        positive = np.clip(self.meta_model_.predict_proba(meta)[:, 1], 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard labels kept for sklearn's estimator validation (calibration)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def build_logistic_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Explainability / logging helpers
# --------------------------------------------------------------------------- #
def shap_ranking(model, X_test: pd.DataFrame, top_n: int = 20) -> list[dict[str, Any]]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_test)
        if isinstance(values, list):
            values = values[-1] if len(values) > 1 else values[0]
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[..., -1]
        mean_abs = np.abs(array).mean(axis=0)
        ranking = sorted(
            zip(X_test.columns, mean_abs), key=lambda pair: pair[1], reverse=True
        )
        return [
            {"feature": str(feature), "mean_abs_shap": float(value)}
            for feature, value in ranking[:top_n]
        ]
    except Exception as exc:  # explainability must not fail the run
        LOGGER.warning("SHAP ranking failed (%s)", exc)
        return []


def _loggable(metrics: dict[str, Any]) -> dict[str, float]:
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            clean[key] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isfinite(number):
                clean[key] = number
    return clean


def _setup_mlflow(mlflow_dir: Path):
    import mlflow

    # MLflow 3.x blocks the file store by default; this project intentionally
    # keeps local runs in ./mlruns (docker-compose maps the same store).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = Path(mlflow_dir).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow, uri


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _concat_frames(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([first, second], axis=0, ignore_index=True)


def _stage_models(
    stages: dict[str, list[str]],
    train_df: pd.DataFrame,
    val_cal: pd.DataFrame,
    test_df: pd.DataFrame,
    lgb_params: dict[str, Any],
) -> tuple[dict[str, dict[str, float]], Any]:
    y_fit = _labels(train_df)
    y_cal = _labels(val_cal)
    y_test = _labels(test_df)
    risk = test_df["cvss_base_score"].to_numpy() if "cvss_base_score" in test_df else None

    ablation: dict[str, dict[str, float]] = {}
    stage3_model = None
    for stage_name in STAGE_NAMES:
        columns = stages[stage_name]
        X_fit = build_matrix(train_df, columns)
        X_cal = build_matrix(val_cal, columns)
        X_test = build_matrix(test_df, columns)
        model = build_model("lightgbm", lgb_params, y_fit)
        model.fit(X_fit, y_fit)
        calibrated = calibrate_model(model, X_cal, y_cal)
        scores = _positive_proba(calibrated, X_test)
        metrics = evaluate_predictions(y_test, scores, risk)
        metrics["n_features"] = len(columns)
        ablation[stage_name] = metrics
        if stage_name == STAGE_NAMES[-1]:
            stage3_model = model
    return ablation, stage3_model


def run_training(
    curated_dir: Path = Path("data/curated"),
    artifact_dir: Path = Path("artifacts/champion"),
    trials: int = 25,
    n_boot: int = 1000,
    skip_mlflow_ui: bool = False,
    mlflow_dir: Path = Path("mlruns"),
    seed: int = 42,
) -> dict[str, Any]:
    curated_dir = Path(curated_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(curated_dir)
    train_df, val_hpo, val_cal, test_df = (splits[name] for name in SPLITS)
    stages = feature_stages(train_df)
    hpo_columns = stages[STAGE_NAMES[1]]

    if "y_180d" not in test_df.columns:
        raise KeyError("split parquet files must contain y_180d")

    X_train = build_matrix(train_df, hpo_columns)
    X_val_hpo = build_matrix(val_hpo, hpo_columns)
    X_val_cal = build_matrix(val_cal, hpo_columns)
    X_test = build_matrix(test_df, hpo_columns)
    y_train, y_val_hpo, y_val_cal, y_test = (
        _labels(train_df),
        _labels(val_hpo),
        _labels(val_cal),
        _labels(test_df),
    )

    LOGGER.info(
        "Rows -> train %s | val_hpo %s | val_cal %s | test %s",
        len(train_df),
        len(val_hpo),
        len(val_cal),
        len(test_df),
    )

    hpo_summary: dict[str, dict[str, Any]] = {}
    best_params: dict[str, dict[str, Any]] = {}
    for kind in MODEL_KINDS:
        study = run_hpo(
            kind, X_train, y_train, X_val_hpo, y_val_hpo, trials, seed=seed
        )
        best_params[kind] = dict(study.best_params)
        hpo_summary[kind] = {
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "best_params": best_params[kind],
        }
        LOGGER.info("HPO %s best val_hpo PR-AUC: %.4f", kind, study.best_value)

    X_fit = _concat_frames(X_train, X_val_hpo)
    y_fit = np.concatenate([y_train, y_val_hpo])

    calibrated_models: dict[str, Any] = {}
    raw_models: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}

    for kind in MODEL_KINDS:
        model = build_model(kind, best_params[kind], y_fit)
        model.fit(X_fit, y_fit)
        raw_models[kind] = model
        calibrated = calibrate_model(model, X_val_cal, y_val_cal)
        calibrated_models[kind] = calibrated
        scores[kind] = _positive_proba(calibrated, X_test)

    stack = StackedEnsemble(hpo_columns, best_params)
    dates_fit = pd.concat(
        [train_df["effective_anchor_date"], val_hpo["effective_anchor_date"]],
        ignore_index=True,
    )
    stack.fit(X_fit, y_fit, dates_fit)
    calibrated_stack = calibrate_model(stack, X_val_cal, y_val_cal)
    calibrated_models["stacked_ensemble"] = calibrated_stack
    scores["stacked_ensemble"] = _positive_proba(calibrated_stack, X_test)

    rule_cvss = (
        pd.to_numeric(test_df["cvss_base_score"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 10.0)
        .to_numpy()
        / 10.0
    )
    rule_epss = (
        pd.to_numeric(test_df["epss"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    )
    scores["Rule_CVSS_Only"] = rule_cvss
    scores["Rule_EPSS_Only"] = rule_epss

    for name, factory in (
        ("LogisticRegression", build_logistic_pipeline),
        ("RandomForest", build_random_forest_pipeline),
    ):
        model = factory()
        model.fit(X_fit, y_fit)
        calibrated = calibrate_model(model, X_val_cal, y_val_cal)
        calibrated_models[name] = calibrated
        scores[name] = _positive_proba(calibrated, X_test)

    risk = (
        pd.to_numeric(test_df["cvss_base_score"], errors="coerce").fillna(0.0).to_numpy()
        if "cvss_base_score" in test_df
        else None
    )
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for name, model_scores in scores.items():
        metrics = evaluate_predictions(y_test, model_scores, risk)
        if n_boot and n_boot > 0:
            metrics["bootstrap_vs_rule_epss"] = paired_bootstrap_delta_pr_auc(
                y_test, model_scores, rule_epss, n_boot=n_boot, seed=seed
            )
            metrics["bootstrap_vs_logistic"] = paired_bootstrap_delta_pr_auc(
                y_test, model_scores, scores["LogisticRegression"], n_boot=n_boot, seed=seed
            )
        metrics_by_model[name] = metrics

    # Champion selection is strictly on validation PR-AUC.
    champion_name = max(MODEL_KINDS, key=lambda k: hpo_summary[k]["best_value"])
    champion_model = calibrated_models[champion_name]
    champion_scores = scores[champion_name]
    p_cal = _positive_proba(champion_model, X_val_cal)
    q_hat = split_conformal_quantile(y_val_cal, p_cal, alpha=ALPHA)
    conformal = {
        "alpha": ALPHA,
        "q_hat": q_hat,
        "calibration_rows": int(len(y_val_cal)),
    }
    conformal.update(conformal_set_metrics(y_test, champion_scores, q_hat))

    ablation, stage3_model = _stage_models(
        stages, train_df, val_cal, test_df, best_params["lightgbm"]
    )

    shap_features = (
        shap_ranking(stage3_model, build_matrix(test_df, stages[STAGE_NAMES[-1]]))
        if stage3_model is not None
        else []
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_NAME,
        "horizon_days": 180,
        "dataset": {
            "rows": {name: int(len(splits[name])) for name in SPLITS},
            "positive_rate": {
                name: round(float(_labels(splits[name]).mean()), 6) for name in SPLITS
            },
        },
        "config": {
            "trials": int(trials),
            "n_boot": int(n_boot),
            "top_fraction": TOP_FRACTION,
            "alpha": ALPHA,
            "seed": int(seed),
        },
        "hpo": hpo_summary,
        "champion": {
            "name": champion_name,
            "selection_metric": "val_hpo_pr_auc",
            "val_hpo_pr_auc": hpo_summary[champion_name]["best_value"],
        },
        "conformal": conformal,
        "ablation": ablation,
        "models": metrics_by_model,
        "shap_top_features": shap_features,
        "feature_stages": {name: list(columns) for name, columns in stages.items()},
    }

    bundle = {
        "model_name": champion_name,
        "model": champion_model,
        # Uncalibrated tree kept for TreeSHAP (the calibrated wrapper is opaque).
        "base_model": raw_models.get(champion_name),
        "feature_columns": hpo_columns,
        "stage2_feature_columns": stages[STAGE_NAMES[1]],
        "stage3_feature_columns": stages[STAGE_NAMES[-1]],
        "tuned_params": best_params,
        "conformal_q_hat": q_hat,
        "conformal_alpha": ALPHA,
        "metrics": metrics_by_model[champion_name],
        "trained_at": summary["generated_at"],
    }
    joblib.dump(bundle, artifact_dir / "champion_bundle.joblib")
    (artifact_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    _log_mlflow(
        summary,
        metrics_by_model,
        ablation,
        artifact_dir,
        mlflow_dir,
        trials,
        n_boot,
        skip_mlflow_ui,
    )

    _print_benchmark(metrics_by_model, champion_name)
    return summary


def _log_mlflow(
    summary: dict[str, Any],
    metrics_by_model: dict[str, dict[str, Any]],
    ablation: dict[str, dict[str, Any]],
    artifact_dir: Path,
    mlflow_dir: Path,
    trials: int,
    n_boot: int,
    skip_mlflow_ui: bool,
) -> None:
    try:
        mlflow, uri = _setup_mlflow(mlflow_dir)
    except Exception as exc:
        LOGGER.warning("MLflow disabled (%s)", exc)
        return

    try:
        with mlflow.start_run(run_name="part3_training"):
            mlflow.log_params(
                {
                    "trials": trials,
                    "n_boot": n_boot,
                    "alpha": ALPHA,
                    "champion": summary["champion"]["name"],
                }
            )
            mlflow.log_metrics(_loggable(summary["conformal"]))
            for name, metrics in metrics_by_model.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_metrics(_loggable(metrics))
            for stage_name, metrics in ablation.items():
                with mlflow.start_run(run_name=stage_name, nested=True):
                    mlflow.log_params({"stage": stage_name, "n_features": metrics["n_features"]})
                    mlflow.log_metrics(_loggable(metrics))
            mlflow.log_artifact(str(Path(artifact_dir) / "metrics_summary.json"))
            mlflow.log_artifact(str(Path(artifact_dir) / "champion_bundle.joblib"))
    except Exception as exc:
        LOGGER.warning("MLflow logging failed (%s)", exc)
        return

    if not skip_mlflow_ui:
        print(f"\nMLflow UI (optional): mlflow ui --backend-store-uri {uri}")


def _print_benchmark(metrics_by_model: dict[str, dict[str, Any]], champion: str) -> None:
    rows = []
    for name, metrics in metrics_by_model.items():
        delta = metrics.get("bootstrap_vs_rule_epss", {})
        rows.append(
            {
                "Model": ("* " if name == champion else "  ") + name,
                "PR-AUC": metrics.get("pr_auc", 0.0),
                "ROC-AUC": metrics.get("roc_auc", 0.0),
                "R@1%": metrics.get("recall_top_1pct", 0.0),
                "P@1%": metrics.get("precision_top_1pct", 0.0),
                "F1@1%": metrics.get("f1_top_1pct", 0.0),
                "RVC@100": metrics.get("rvc_100", 0.0),
                "RVC@500": metrics.get("rvc_500", 0.0),
                "Brier": metrics.get("brier", 0.0),
                "ECE": metrics.get("ece_10", 0.0),
                "dPR vs EPSS [95% CI]": (
                    f"{delta.get('delta_mean', 0):+.3f} "
                    f"[{delta.get('ci_low', 0):+.3f},{delta.get('ci_high', 0):+.3f}]"
                ),
                "p": f"{delta.get('p_value_one_sided', 1.0):.3f}",
            }
        )
    table = pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    print("\nHELD-OUT TEST BENCHMARK (* = champion by val_hpo PR-AUC)")
    print("=" * 120)
    print(table)
    print("=" * 120)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark AegisKEV models.")
    parser.add_argument("--curated-dir", type=Path, default=Path("data/curated"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/champion")
    )
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--skip-mlflow-ui", action="store_true")
    parser.add_argument("--mlflow-dir", type=Path, default=Path("mlruns"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    run_training(
        curated_dir=args.curated_dir,
        artifact_dir=args.artifact_dir,
        trials=args.trials,
        n_boot=args.n_boot,
        skip_mlflow_ui=args.skip_mlflow_ui,
        mlflow_dir=args.mlflow_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
