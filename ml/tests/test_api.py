"""FastAPI inference API tests (no live Postgres required)."""

from __future__ import annotations

import json
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.main import create_app
from ml.training.train_pipeline import BASE_FEATURES, STAGE2_FEATURES

FEATURE_COLUMNS = list(BASE_FEATURES) + list(STAGE2_FEATURES)

SAMPLE = {
    "cve_id": "CVE-2021-44228",
    "cvss_base_score": 10.0,
    "cvss_exploitability_score": 3.9,
    "cvss_impact_score": 6.0,
    "is_network_av": True,
    "is_low_ac": True,
    "is_priv_none": True,
    "is_ui_none": True,
    "scope_changed": True,
    "cwe_count": 2,
    "cpe_config_count": 1,
    "cpe_vendor_count": 1,
    "reference_count": 9,
    "has_vendor_advisory_ref": True,
    "has_patch_tag": False,
    "has_exploit_tag": True,
    "epss": 0.94,
    "epss_percentile": 0.99,
}


def _make_bundle(tmp_path) -> "Path":
    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(
        rng.normal(size=(80, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    labels = rng.integers(0, 2, len(matrix))
    model = Pipeline(
        [("scaler", StandardScaler()), ("logistic", LogisticRegression(max_iter=500))]
    ).fit(matrix, labels)
    bundle = {
        "model_name": "test_logistic",
        "model": model,
        "base_model": model,
        "feature_columns": FEATURE_COLUMNS,
        "conformal_q_hat": 0.2,
        "conformal_alpha": 0.10,
    }
    path = tmp_path / "champion_bundle.joblib"
    joblib.dump(bundle, path)
    return path


def _make_metrics(tmp_path) -> "Path":
    payload = {
        "experiment": "AegisKEV_180d_Exploit_Prediction",
        "champion": {"name": "lightgbm"},
        "models": {
            "lightgbm": {"pr_auc": 0.42, "ece_10": 0.05},
            "Rule_EPSS_Only": {"pr_auc": 0.30, "ece_10": 0.11},
        },
        "ablation": {
            "Stage_1_Base_NVD_EPSS": {"pr_auc": 0.31, "n_features": 19},
            "Stage_2_Plus_Causal_Priors_Race": {"pr_auc": 0.40, "n_features": 26},
        },
    }
    path = tmp_path / "metrics_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_manifest(tmp_path) -> "Path":
    payload = {
        "censoring_cutoff": "2025-01-01T00:00:00",
        "row_counts": {"train": 10, "val_hpo": 4, "val_cal": 3, "test": 2},
        "unmatured_rows": 1,
        "date_ranges": {
            "train": {"min": "2021-01-01", "max": "2022-01-01"},
            "val_hpo": {"min": "2022-01-02", "max": "2022-06-01"},
            "val_cal": {"min": "2022-06-02", "max": "2022-09-01"},
            "test": {"min": "2022-09-02", "max": "2023-01-01"},
        },
    }
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        bundle_path=_make_bundle(tmp_path),
        metrics_path=_make_metrics(tmp_path),
        manifest_path=_make_manifest(tmp_path),
        unmatured_path=tmp_path / "unmatured_recent.parquet",
        db_url="postgresql://bad:bad@127.0.0.1:1/nope",
        sqlite_path=tmp_path / "prediction_logs.sqlite3",
        jsonl_path=tmp_path / "prediction_logs.jsonl",
    )
    with TestClient(app) as test_client:
        yield test_client, tmp_path


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_default_cors_allows_screenshot_dev_origin(tmp_path, monkeypatch):
    monkeypatch.delenv("AEGIS_CORS_ORIGINS", raising=False)
    app = create_app(
        bundle_path=_make_bundle(tmp_path),
        metrics_path=_make_metrics(tmp_path),
        sqlite_path=tmp_path / "logs.sqlite3",
        jsonl_path=tmp_path / "logs.jsonl",
    )
    with TestClient(app) as test_client:
        response = test_client.get(
            "/health", headers={"Origin": "http://127.0.0.1:3100"}
        )
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3100"


def test_ready_reports_loaded_model(client):
    test_client, _ = client
    response = test_client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "test_logistic"


def test_ready_returns_503_without_bundle(tmp_path):
    app = create_app(
        bundle_path=tmp_path / "missing.joblib",
        metrics_path=tmp_path / "metrics.json",
        db_url="postgresql://bad:bad@127.0.0.1:1/nope",
        sqlite_path=tmp_path / "logs.sqlite3",
        jsonl_path=tmp_path / "logs.jsonl",
    )
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False
        predict = test_client.post("/api/v1/predict", json=SAMPLE)
        assert predict.status_code == 503


def test_predict_returns_full_contract(client):
    test_client, _ = client
    response = test_client.post("/api/v1/predict", json=SAMPLE)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cve_id"] == SAMPLE["cve_id"]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_tier"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert body["conformal_set"] in {"{0}", "{1}", "{0,1}"}
    assert isinstance(body["requires_manual_analyst_review"], bool)
    assert len(body["top_drivers"]) <= 8
    for driver in body["top_drivers"]:
        assert driver["direction"] in {"increases_risk", "decreases_risk"}
    assert "Forecast only" in body["caveat"]
    assert body["scored_at"]


def test_predict_rejects_invalid_cve_id(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/predict", json={**SAMPLE, "cve_id": "CVE-2021-123"}
    )
    assert response.status_code == 422


def test_batch_scoring_and_limit(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/predict/batch", json={"items": [SAMPLE, {**SAMPLE, "cve_id": "CVE-2024-1234"}]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [item["cve_id"] for item in body["predictions"]] == [
        "CVE-2021-44228",
        "CVE-2024-1234",
    ]

    too_many = test_client.post(
        "/api/v1/predict/batch", json={"items": [SAMPLE] * 101}
    )
    assert too_many.status_code == 422


def test_benchmark_serves_verified_metrics(client):
    test_client, _ = client
    response = test_client.get("/api/v1/metrics/benchmark")
    assert response.status_code == 200
    body = response.json()
    assert body["experiment"] == "AegisKEV_180d_Exploit_Prediction"
    assert "lightgbm" in body["models"]


def test_recent_triage_scores_ranked_unmatured_rows(client):
    test_client, tmp_path = client
    pd.DataFrame(
        [
            {
                **SAMPLE,
                "cve_id": "CVE-2024-1111",
                "published_date": pd.Timestamp("2024-01-01"),
                "epss": 0.2,
                "cvss_base_score": 8.0,
            },
            {
                **SAMPLE,
                "cve_id": "CVE-2024-2222",
                "published_date": pd.Timestamp("2024-02-01"),
                "epss": 0.9,
                "cvss_base_score": 7.0,
            },
        ]
    ).to_parquet(tmp_path / "unmatured_recent.parquet", index=False)

    response = test_client.get("/api/v1/triage/recent?limit=1")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_unmatured"] == 2
    assert body["cutoff_date"] == "2025-01-01T00:00:00"
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["cve_id"] == "CVE-2024-2222"
    assert item["epss"] == pytest.approx(0.9)
    assert 0.0 <= item["probability"] <= 1.0
    assert item["conformal_set"] in {"{0}", "{1}", "{0,1}"}
    assert isinstance(item["requires_manual_analyst_review"], bool)
    assert item["features"]["epss"] == pytest.approx(0.9)


def test_recent_triage_is_empty_when_parquet_missing(client):
    test_client, _ = client
    response = test_client.get("/api/v1/triage/recent")
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total_unmatured"] == 0


def test_markdown_report_downloads_from_metrics_and_manifest(client):
    test_client, _ = client
    response = test_client.get("/api/v1/reports/markdown")
    assert response.status_code == 200, response.text
    assert "text/markdown" in response.headers["content-type"]
    assert "attachment; filename=AegisKEV_Evaluation_Report.md" in response.headers[
        "content-disposition"
    ]
    assert "Chronological splits" in response.text
    assert "Held-out test benchmark" in response.text
    assert "Three-stage feature ablation" in response.text
    assert "0.420" in response.text


def test_benchmark_503_when_metrics_missing(tmp_path):
    app = create_app(
        bundle_path=_make_bundle(tmp_path),
        metrics_path=tmp_path / "missing_metrics.json",
        db_url="postgresql://bad:bad@127.0.0.1:1/nope",
        sqlite_path=tmp_path / "logs.sqlite3",
        jsonl_path=tmp_path / "logs.jsonl",
    )
    with TestClient(app) as test_client:
        response = test_client.get("/api/v1/metrics/benchmark")
        assert response.status_code == 503
        assert "train_pipeline" in response.json()["detail"]


def test_predictions_fall_back_to_sqlite_when_postgres_offline(client):
    test_client, tmp_path = client
    response = test_client.post("/api/v1/predict", json=SAMPLE)
    assert response.status_code == 200

    database = tmp_path / "prediction_logs.sqlite3"
    assert database.exists()
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT cve_id, predicted_probability, risk_tier FROM prediction_logs"
        ).fetchall()
    assert rows
    assert rows[0][0] == "CVE-2021-44228"
    assert 0.0 <= rows[0][1] <= 1.0
