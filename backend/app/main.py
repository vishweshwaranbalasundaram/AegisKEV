"""AegisKEV production inference API.

Loads the calibrated champion bundle once, serves point-in-time predictions,
exposes verified benchmark metrics, and asynchronously logs every prediction to
PostgreSQL with a SQLite/JSONL fallback for local development.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from backend.app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CveFeatures,
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
    RecentTriageResponse,
    ShapDriver,
    TriageItem,
)

LOGGER = logging.getLogger("aegiskev.api")

CAVEAT = (
    "Forecast only. The 180-day KEV exploitation probability is not a "
    "vulnerability severity score or a guarantee of exploitation. EPSS is "
    "anchored at NVD publication + 1 day (not the latest snapshot), conformal "
    "sets may be ambiguous, and HIGH/CRITICAL or ambiguous cases require "
    "manual analyst review."
)

CRITICAL_THRESHOLD = 0.50
HIGH_THRESHOLD = 0.25
MEDIUM_THRESHOLD = 0.10

TOP_DRIVERS = 8


# --------------------------------------------------------------------------- #
# Persistence (PostgreSQL with SQLite/JSONL fallback)
# --------------------------------------------------------------------------- #
POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    cve_id TEXT NOT NULL,
    predicted_probability DOUBLE PRECISION NOT NULL,
    risk_tier TEXT NOT NULL,
    conformal_set TEXT NOT NULL,
    requires_review BOOLEAN NOT NULL,
    model_name TEXT NOT NULL,
    shap_drivers JSONB,
    request_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cve_id TEXT NOT NULL,
    predicted_probability REAL NOT NULL,
    risk_tier TEXT NOT NULL,
    conformal_set TEXT NOT NULL,
    requires_review INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    shap_drivers TEXT,
    request_payload TEXT,
    created_at TEXT NOT NULL
)
"""


class PredictionLogger:
    """Best-effort async audit log: Postgres -> SQLite -> JSONL."""

    def __init__(
        self,
        db_url: str | None,
        sqlite_path: Path = Path("data/prediction_logs.sqlite3"),
        jsonl_path: Path = Path("data/prediction_logs.jsonl"),
    ) -> None:
        self.db_url = db_url
        self.sqlite_path = Path(sqlite_path)
        self.jsonl_path = Path(jsonl_path)
        self._postgres_available = bool(db_url)

    def ensure_schema(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(SQLITE_DDL)
            connection.commit()
        if self._postgres_available:
            try:
                with self._connect_postgres() as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(POSTGRES_DDL)
                    connection.commit()
            except Exception as exc:  # noqa: BLE001 - offline dev is expected
                self._postgres_available = False
                LOGGER.warning("Postgres audit log unavailable (%s); using SQLite", exc)

    def _connect_postgres(self):
        import psycopg

        return psycopg.connect(self.db_url, connect_timeout=2)

    def log(self, record: dict[str, Any]) -> str:
        if self._postgres_available:
            try:
                self._write_postgres(record)
                return "postgres"
            except Exception as exc:  # noqa: BLE001
                self._postgres_available = False
                LOGGER.warning("Postgres insert failed (%s); falling back", exc)
        try:
            self._write_sqlite(record)
            return "sqlite"
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("SQLite insert failed (%s); writing JSONL", exc)
            self._write_jsonl(record)
            return "jsonl"

    def _write_postgres(self, record: dict[str, Any]) -> None:
        from psycopg.types.json import Jsonb

        with self._connect_postgres() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO prediction_logs
                        (cve_id, predicted_probability, risk_tier, conformal_set,
                         requires_review, model_name, shap_drivers, request_payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record["cve_id"],
                        record["predicted_probability"],
                        record["risk_tier"],
                        record["conformal_set"],
                        record["requires_review"],
                        record["model_name"],
                        Jsonb(record["shap_drivers"]),
                        Jsonb(record["request_payload"]),
                    ),
                )
            connection.commit()

    def _write_sqlite(self, record: dict[str, Any]) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT INTO prediction_logs
                    (cve_id, predicted_probability, risk_tier, conformal_set,
                     requires_review, model_name, shap_drivers, request_payload,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["cve_id"],
                    float(record["predicted_probability"]),
                    record["risk_tier"],
                    record["conformal_set"],
                    int(bool(record["requires_review"])),
                    record["model_name"],
                    json.dumps(record["shap_drivers"]),
                    json.dumps(record["request_payload"]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Feature derivation + scoring
# --------------------------------------------------------------------------- #
def _logit(value: float | None) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _item_to_row(item: CveFeatures) -> dict[str, Any]:
    data = item.model_dump()
    epss = data.get("epss")
    cvss = data.get("cvss_base_score")
    impact = data.get("cvss_impact_score")
    exploitability = data.get("cvss_exploitability_score")
    cvss_norm = (float(cvss) / 10.0) if cvss is not None else 0.0

    data["epss_missing_flag"] = (
        data["epss_missing_flag"]
        if data.get("epss_missing_flag") is not None
        else (0 if epss is not None else 1)
    )
    data["epss_logit"] = (
        data["epss_logit"] if data.get("epss_logit") is not None else _logit(epss)
    )
    data["epss_same_day_pct"] = (
        data["epss_same_day_pct"]
        if data.get("epss_same_day_pct") is not None
        else (epss if epss is not None else float("nan"))
    )
    data["epss_minus_cvss_norm"] = (
        data["epss_minus_cvss_norm"]
        if data.get("epss_minus_cvss_norm") is not None
        else ((epss if epss is not None else float("nan")) - cvss_norm)
    )
    data["exploit_without_patch_race"] = (
        data["exploit_without_patch_race"]
        if data.get("exploit_without_patch_race") is not None
        else int(bool(data.get("has_exploit_tag"))) - int(bool(data.get("has_patch_tag")))
    )
    data["zero_click_network_flaw"] = (
        data["zero_click_network_flaw"]
        if data.get("zero_click_network_flaw") is not None
        else int(
            bool(data.get("is_network_av"))
            and bool(data.get("is_low_ac"))
            and bool(data.get("is_priv_none"))
            and bool(data.get("is_ui_none"))
        )
    )
    data["exploit_to_impact_ratio"] = (
        data["exploit_to_impact_ratio"]
        if data.get("exploit_to_impact_ratio") is not None
        else (
            float(exploitability) / float(impact)
            if impact and exploitability is not None and float(impact) > 0
            else float("nan")
        )
    )
    if data.get("cwe_kev_prior") is None:
        data["cwe_kev_prior"] = 0.0
    if data.get("primary_vendor_kev_prior") is None:
        data["primary_vendor_kev_prior"] = 0.0
    return data


def _feature_matrix(items: Sequence[CveFeatures], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame([_item_to_row(item) for item in items])
    matrix = frame.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce")
    return matrix.astype("float64")


def _shap_matrix(
    bundle: dict[str, Any], matrix: pd.DataFrame
) -> tuple[np.ndarray | None, float | None]:
    base_model = bundle.get("base_model")
    if base_model is None:
        return None, None
    try:
        import shap

        explainer = shap.TreeExplainer(base_model)
        values = explainer.shap_values(matrix)
        expected = np.asarray(explainer.expected_value).reshape(-1)
        if isinstance(values, list):
            values = values[-1] if len(values) > 1 else values[0]
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[..., -1]
        base_value = float(expected[-1]) if expected.size else None
        return array.reshape(len(matrix), -1), base_value
    except Exception as exc:  # noqa: BLE001 - explainability is best-effort
        LOGGER.info("TreeSHAP unavailable for %s (%s)", bundle.get("model_name"), exc)
        return None, None


def _conformal_set(probability: float, q_hat: float) -> str:
    include_one = (1.0 - probability) <= q_hat + 1e-12
    include_zero = probability <= q_hat + 1e-12
    if include_zero and include_one:
        return "{0,1}"
    if include_one:
        return "{1}"
    if include_zero:
        return "{0}"
    return "{0,1}"  # empty set: abstain conservatively


def _risk_tier(probability: float) -> str:
    if probability >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def _score(
    bundle: dict[str, Any], items: Sequence[CveFeatures]
) -> tuple[list[PredictionResponse], list[dict[str, Any]]]:
    columns = bundle["feature_columns"]
    matrix = _feature_matrix(items, columns)
    probabilities = bundle["model"].predict_proba(matrix)[:, 1]
    q_hat = float(bundle.get("conformal_q_hat", 1.0))
    contributions, shap_base_value = _shap_matrix(bundle, matrix)

    responses: list[PredictionResponse] = []
    records: list[dict[str, Any]] = []
    for index, (item, probability) in enumerate(zip(items, probabilities)):
        probability = float(min(max(probability, 0.0), 1.0))
        conformal = _conformal_set(probability, q_hat)
        tier = _risk_tier(probability)
        requires_review = bool(
            conformal == "{0,1}" or tier in {"CRITICAL", "HIGH"}
        )

        all_drivers: list[ShapDriver] = []
        if contributions is not None and contributions.shape[0] > index:
            row = contributions[index]
            ranking = sorted(
                range(min(len(columns), len(row))),
                key=lambda position: abs(float(row[position])),
                reverse=True,
            )
            all_drivers = [
                ShapDriver(
                    feature=str(columns[position]),
                    contribution=float(row[position]),
                    direction=(
                        "increases_risk" if float(row[position]) >= 0 else "decreases_risk"
                    ),
                )
                for position in ranking
            ]
        drivers = all_drivers[:TOP_DRIVERS]

        scored_at = datetime.now(timezone.utc).isoformat()
        response = PredictionResponse(
            cve_id=item.cve_id,
            probability=probability,
            risk_tier=tier,
            conformal_set=conformal,
            requires_manual_analyst_review=requires_review,
            top_drivers=drivers,
            shap_base_value=shap_base_value,
            shap_values=all_drivers,
            model_name=str(bundle.get("model_name", "unknown")),
            caveat=CAVEAT,
            scored_at=scored_at,
        )
        responses.append(response)
        records.append(
            {
                "cve_id": item.cve_id,
                "predicted_probability": probability,
                "risk_tier": tier,
                "conformal_set": conformal,
                "requires_review": requires_review,
                "model_name": response.model_name,
                "shap_drivers": [driver.model_dump() for driver in drivers],
                "request_payload": item.model_dump(mode="json"),
            }
        )
    return responses, records


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def _load_bundle(app: FastAPI) -> dict[str, Any] | None:
    if app.state.bundle is not None:
        return app.state.bundle
    try:
        app.state.bundle = joblib.load(app.state.bundle_path)
        app.state.bundle_error = None
    except Exception as exc:  # noqa: BLE001 - /ready reports the reason
        app.state.bundle_error = str(exc)
    return app.state.bundle


def create_app(
    bundle_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    unmatured_path: Path | str | None = None,
    db_url: str | None = None,
    sqlite_path: Path | str | None = None,
    jsonl_path: Path | str | None = None,
) -> FastAPI:
    bundle_path = Path(
        bundle_path
        or os.getenv(
            "AEGIS_BUNDLE_PATH", "artifacts/champion/champion_bundle.joblib"
        )
    )
    metrics_path = Path(
        metrics_path
        or os.getenv(
            "AEGIS_METRICS_PATH", "artifacts/champion/metrics_summary.json"
        )
    )
    manifest_path = Path(
        manifest_path
        or os.getenv("AEGIS_MANIFEST_PATH", "data/curated/dataset_manifest.json")
    )
    unmatured_path = Path(
        unmatured_path
        or os.getenv("AEGIS_UNMATURED_PATH", "data/curated/unmatured_recent.parquet")
    )
    db_url = db_url if db_url is not None else os.getenv("DATABASE_URL", "")
    sqlite_path = Path(
        sqlite_path
        or os.getenv("AEGIS_SQLITE_PATH", "data/prediction_logs.sqlite3")
    )
    jsonl_path = Path(
        jsonl_path or os.getenv("AEGIS_JSONL_PATH", "data/prediction_logs.jsonl")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_bundle(app)
        app.state.logger = PredictionLogger(
            db_url=db_url or None,
            sqlite_path=sqlite_path,
            jsonl_path=jsonl_path,
        )
        app.state.logger.ensure_schema()
        yield

    app = FastAPI(
        title="AegisKEV Inference API",
        version="0.4.0",
        description="Point-in-time 180-day CISA KEV exploitation forecasting.",
        lifespan=lifespan,
    )
    app.state.bundle_path = bundle_path
    app.state.metrics_path = metrics_path
    app.state.manifest_path = manifest_path
    app.state.unmatured_path = unmatured_path
    app.state.bundle = None
    app.state.bundle_error = None
    app.state.logger = None

    origins = [
        origin.strip()
        for origin in os.getenv(
            "AEGIS_CORS_ORIGINS",
            (
                "http://localhost:3000,http://127.0.0.1:3000,"
                "http://localhost:3100,http://127.0.0.1:3100"
            ),
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="aegiskev-api")

    @app.get("/ready", response_model=ReadyResponse)
    def ready(request: Request):
        bundle = _load_bundle(request.app)
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content=ReadyResponse(
                    status="not_ready",
                    model_loaded=False,
                    detail=request.app.state.bundle_error or "champion bundle missing",
                ).model_dump(),
            )
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            model_name=str(bundle.get("model_name", "unknown")),
        )

    def _require_bundle(request: Request) -> dict[str, Any]:
        bundle = _load_bundle(request.app)
        if bundle is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Champion bundle not loaded. Train it with "
                    "`python -m ml.training.train_pipeline`."
                ),
            )
        return bundle

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    def predict(
        item: CveFeatures, request: Request, background: BackgroundTasks
    ) -> PredictionResponse:
        bundle = _require_bundle(request)
        predictions, records = _score(bundle, [item])
        background.add_task(request.app.state.logger.log, records[0])
        return predictions[0]

    @app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictRequest, request: Request, background: BackgroundTasks
    ) -> BatchPredictionResponse:
        bundle = _require_bundle(request)
        predictions, records = _score(bundle, payload.items)
        for record in records:
            background.add_task(request.app.state.logger.log, record)
        return BatchPredictionResponse(count=len(predictions), predictions=predictions)

    @app.get("/api/v1/metrics/benchmark")
    def benchmark(request: Request):
        path = Path(request.app.state.metrics_path)
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Benchmark metrics not generated. Run "
                    "`python -m ml.training.train_pipeline` to populate "
                    "artifacts/champion/metrics_summary.json."
                ),
            )
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
            manifest_path = Path(request.app.state.manifest_path)
            if manifest_path.exists():
                metrics["dataset_manifest"] = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            return metrics
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail=f"Benchmark metrics are corrupt: {exc}"
            ) from exc

    @app.get("/api/v1/triage/recent", response_model=RecentTriageResponse)
    def recent_triage(request: Request, limit: int = 25) -> RecentTriageResponse:
        manifest_path = Path(request.app.state.manifest_path)
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=500, detail=f"Dataset manifest is corrupt: {exc}"
                ) from exc
        cutoff_date = manifest.get("censoring_cutoff")
        path = Path(request.app.state.unmatured_path)
        if not path.exists():
            return RecentTriageResponse(
                total_unmatured=0, cutoff_date=cutoff_date, items=[]
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            LOGGER.exception("Unable to read unmatured triage parquet")
            raise HTTPException(
                status_code=500, detail=f"Unable to read unmatured CVEs: {exc}"
            ) from exc

        total = len(frame)
        safe_limit = max(1, min(100, limit))
        if "epss" not in frame.columns:
            frame["epss"] = np.nan
        if "cvss_base_score" not in frame.columns:
            frame["cvss_base_score"] = np.nan
        ranked = frame.assign(
            _epss_rank=pd.to_numeric(frame["epss"], errors="coerce"),
            _cvss_rank=pd.to_numeric(frame["cvss_base_score"], errors="coerce"),
        ).sort_values(
            ["_epss_rank", "_cvss_rank"],
            ascending=[False, False],
            na_position="last",
            kind="mergesort",
        ).head(safe_limit)

        allowed_fields = set(CveFeatures.model_fields)
        selected: list[tuple[CveFeatures, dict[str, Any]]] = []
        for raw in ranked.to_dict(orient="records"):
            feature_values: dict[str, Any] = {}
            for name in allowed_fields:
                value = raw.get(name)
                if value is None or pd.isna(value):
                    continue
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (pd.Timestamp, datetime)):
                    value = value.isoformat()
                feature_values[name] = value
            try:
                features = CveFeatures.model_validate(feature_values)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid feature row in unmatured CVE data: {exc}",
                ) from exc
            selected.append((features, raw))

        bundle = _require_bundle(request)
        predictions, _ = _score(bundle, [features for features, _ in selected])
        items = []
        for prediction, (features, raw) in zip(predictions, selected):
            payload = prediction.model_dump()
            items.append(
                TriageItem(
                    **payload,
                    published_date=(
                        raw.get("published_date").isoformat()
                        if isinstance(raw.get("published_date"), (pd.Timestamp, datetime))
                        else str(raw.get("published_date"))
                        if raw.get("published_date") is not None
                        else None
                    ),
                    cvss_base_score=features.cvss_base_score,
                    epss=features.epss,
                    top_shap_driver=(
                        prediction.top_drivers[0].feature
                        if prediction.top_drivers
                        else None
                    ),
                    features=features,
                )
            )
        return RecentTriageResponse(
            total_unmatured=total, cutoff_date=cutoff_date, items=items
        )

    @app.get("/api/v1/reports/markdown")
    def markdown_report(request: Request) -> Response:
        metrics_path = Path(request.app.state.metrics_path)
        manifest_path = Path(request.app.state.manifest_path)
        if not metrics_path.exists() or not manifest_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Evaluation report artifacts are incomplete. Generate "
                    "metrics_summary.json and dataset_manifest.json first."
                ),
            )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail=f"Evaluation artifacts are corrupt: {exc}"
            ) from exc

        def number(value: Any, digits: int = 3) -> str:
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"

        def count(value: Any) -> str:
            return f"{value:,}" if isinstance(value, int) else "—"

        split_counts = manifest.get("row_counts", {})
        matured_count = sum(
            value for value in split_counts.values() if isinstance(value, int)
        )
        unmatured_count = manifest.get("unmatured_rows", 0)
        total_count = matured_count + (
            unmatured_count if isinstance(unmatured_count, int) else 0
        )
        models = metrics.get("models", {})
        model_lines = [
            "| Model | PR-AUC | 95% paired-bootstrap CI (Δ vs EPSS) | Recall@Top-1% | RVC@500 | ECE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, result in models.items():
            bootstrap = result.get("bootstrap_vs_rule_epss", {})
            interval = (
                f"[{number(bootstrap.get('ci_low'))}, {number(bootstrap.get('ci_high'))}]"
                if bootstrap
                else "—"
            )
            model_lines.append(
                f"| {name} | {number(result.get('pr_auc'))} | {interval} | "
                f"{number(result.get('recall_top_1pct'))} | "
                f"{number(result.get('rvc_500'))} | {number(result.get('ece_10'))} |"
            )

        stage_names = (
            "Stage_1_Base_NVD_EPSS",
            "Stage_2_Plus_Causal_Priors_Race",
            "Stage_3_Plus_Text_Embeddings",
        )
        stage_lines = [
            "| Feature stage | PR-AUC | Recall@Top-1% | RVC@500 | Features |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in stage_names:
            stage = metrics.get("ablation", {}).get(name)
            if stage is not None:
                stage_lines.append(
                    f"| {name.replace('_', ' ')} | {number(stage.get('pr_auc'))} | "
                    f"{number(stage.get('recall_top_1pct'))} | "
                    f"{number(stage.get('rvc_500'))} | {count(stage.get('n_features'))} |"
                )

        shap_lines = [
            "| Rank | Feature | Mean absolute SHAP |",
            "|---:|---|---:|",
        ]
        for index, feature in enumerate(metrics.get("shap_top_features", [])[:15], 1):
            shap_lines.append(
                f"| {index} | {feature.get('feature', '—')} | "
                f"{number(feature.get('mean_abs_shap'))} |"
            )
        date_ranges = manifest.get("date_ranges", {})
        split_lines = [
            "| Chronological split | Rows | Date range |",
            "|---|---:|---|",
        ]
        for name in ("train", "val_hpo", "val_cal", "test"):
            date_range = date_ranges.get(name, {})
            split_lines.append(
                f"| {name} | {count(split_counts.get(name))} | "
                f"{date_range.get('min', '—')} to {date_range.get('max', '—')} |"
            )

        conformal = metrics.get("conformal", {})
        report = "\n".join(
            [
                "# AegisKEV Technical Evaluation Report",
                "",
                f"Generated from the champion metrics artifact (`{metrics.get('generated_at', '—')}`).",
                "",
                "## Dataset scale",
                "",
                f"- Curated matured rows: **{count(matured_count)}**",
                f"- Right-censored rows: **{count(unmatured_count)}**",
                f"- Total curated and right-censored CVEs: **{count(total_count)}**",
                f"- Held-out test rows: **{count(split_counts.get('test'))}**",
                f"- Censoring cutoff: **{manifest.get('censoring_cutoff', '—')}**",
                "",
                "## Chronological splits",
                "",
                *split_lines,
                "",
                "## Held-out test benchmark",
                "",
                f"Champion selection was based on `{metrics.get('champion', {}).get('selection_metric', '—')}`; the test split was held out.",
                "Bootstrap intervals below are paired 95% intervals for PR-AUC delta versus the EPSS-only baseline, as recorded in the metrics artifact.",
                "",
                *model_lines,
                "",
                "## Three-stage feature ablation",
                "",
                *stage_lines,
                "",
                "## Split-conformal coverage",
                "",
                f"- Alpha: **{number(conformal.get('alpha'))}**",
                f"- Empirical coverage: **{number(conformal.get('coverage'), 4)}**",
                f"- Ambiguous prediction-set fraction: **{number(conformal.get('ambiguous_fraction'), 4)}**",
                f"- Calibration rows: **{count(conformal.get('calibration_rows'))}**",
                "",
                "## Global TreeSHAP drivers",
                "",
                *shap_lines,
                "",
                "Forecast only; probabilities are not severity scores or guarantees of exploitation.",
                "",
            ]
        )
        return Response(
            content=report,
            media_type="text/markdown",
            headers={
                "Content-Disposition": "attachment; filename=AegisKEV_Evaluation_Report.md"
            },
        )

    return app


app = create_app()
