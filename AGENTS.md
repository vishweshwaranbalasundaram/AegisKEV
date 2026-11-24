# AGENTS.md

AegisKEV — point-in-time 180-day CISA KEV exploitation forecasting.

## Layout

| Path | Purpose |
| --- | --- |
| `ml/ingestion/sync_data.py` | KEV / NVD / EPSS ingestion CLI + pure parsing helpers |
| `ml/features/advanced_features.py` | causal group priors, domain interactions, BGE+PCA, TF-IDF+SVD |
| `ml/features/build_dataset.py` | join → target → right-censor → chronological split → parquet + manifest |
| `ml/training/train_pipeline.py` | HPO, calibration, conformal, ablation, SHAP, MLflow, `artifacts/champion/` bundle |
| `backend/app/main.py` | FastAPI inference API + PostgreSQL/SQLite/JSONL audit log |
| `backend/app/schemas.py` | Pydantic v2 request/response contracts |
| `frontend/` | Next.js App Router SOC workbench (triage, benchmark, methodology) |
| `ml/tests/` | offline Pytest suite (ingestion + leakage + training + API) |
| `Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` | api / frontend / postgres / mlflow orchestration |
| `.github/workflows/ci.yml` | pytest + Next.js typecheck/build |
| `data/raw/`, `data/curated/`, `artifacts/` | git-ignored datasets, splits, model artifacts |

## Commands

Run from the repo root (`pyproject.toml` sets `pythonpath = ["."]`).

- All tests: `python -m pytest -v ml/tests/`
- One test: `python -m pytest -v "ml/tests/test_leakage.py::test_causal_prior_reveals_on_maturation_day_not_before"`
- Ingest: `python -m ml.ingestion.sync_data --years 2021-2026 --workers 8`
- Build dataset: `python -m ml.features.build_dataset`
- Dry run without the ML stack: add `--skip-embeddings --skip-tfidf`
- Train + benchmark: `python -m ml.training.train_pipeline --trials 25 --n-boot 1000`
  (`--curated-dir`, `--artifact-dir`, `--trials`, `--n-boot`, `--skip-mlflow-ui`, `--mlflow-dir`, `--seed`)
- API: `uvicorn backend.app.main:app --reload --port 8000`
- Workbench: `cd frontend && npm install && npm run dev` (`npm run build` for production)
- Full stack: `docker compose up --build`

## Environment reality

- Local interpreter is Python 3.12.10. Installed: `pandas`, `numpy`, `pyarrow`, `requests`, `pytest`, `scikit-learn` 1.9.1, `lightgbm` 4.7.0, `xgboost` 3.4.1, `catboost` 1.2.10, `optuna` 5.1.0, `shap` 0.54.1, `mlflow` 3.18.0, `scipy`, `joblib`, `psycopg`, FastAPI/uvicorn/pydantic.
- `sentence-transformers` and `torch` are **not** installed. `build_dataset` imports them lazily, so only the `--skip-*` path runs locally; never import them at module top level.
- Without `torch`, Optuna's `GPSampler` may be unavailable; `make_sampler` logs and falls back to `TPESampler` automatically.
- MLflow 3.x blocks the plain `./mlruns` file store by default; `_setup_mlflow` sets `MLFLOW_ALLOW_FILE_STORE=true`. `docker-compose` runs a separate MLflow server on port 5000.

## Time / leakage contract (do not weaken)

- **EPSS anchor = NVD `published` + 1 day** (`+2` fallback) for non-rejected CVEs published on/after `2021-04-14`. Never KEV `dateAdded` — it leaks post-KEV EPSS spikes and drops negatives. `fetch_epss_anchored` rejects frames without `published`.
- Target: `y_180d = 1` iff `anchor < date_added <= anchor + 180d`. Rows with `date_added <= anchor` are excluded (already in KEV at triage).
- Right-censoring: `kev_cutoff = max(date_added)`; anchors after `kev_cutoff - 180d` go to `data/curated/unmatured_recent.parquet`, never into labeled splits.
- Causal priors reveal a CVE's label only when `current_day >= anchor + 180d`; same-day rows are scored as one batch. Priors are computed on matured rows **before** splitting.
- Text PCA/TF-IDF/SVD are fit on `train` only, then applied to the other splits.
- Splits: `train <= 2024-06-30`, `val_hpo 2024-07-01..2024-11-30`, `val_cal 2024-12-01..2025-03-16`, `test >= 2025-03-17`. If any window is empty (reduced year subset), a quantile fallback keeps four contiguous blocks.
- `cisa*` columns and `date_added` must never reach a feature frame; `finalize_features` drops them and `assert_no_cisa_columns` enforces it.

## Outputs

- `data/raw/kev/kev_catalog.parquet`, `data/raw/nvd/nvd_parsed.parquet`, `data/raw/epss_anchored.parquet` (has `anchor_date`).
- `data/curated/{train,val_hpo,val_cal,test}.parquet`, `unmatured_recent.parquet`, `dataset_manifest.json` (row counts, prevalence, date ranges, SHA-256 per file).
- `artifacts/preprocessors/text_pipeline.joblib` (TF-IDF+SVD backup, fit on train).
- `artifacts/champion/champion_bundle.joblib` — calibrated champion, uncalibrated `base_model` for TreeSHAP, `feature_columns`, `conformal_q_hat`, tuned params, test metrics.
- `artifacts/champion/metrics_summary.json` — per-model test metrics, paired-bootstrap deltas vs `Rule_EPSS_Only`/`LogisticRegression`, conformal coverage/ambiguity, 3-stage ablation, SHAP ranking, dataset manifest.
- `data/prediction_logs.sqlite3` / `data/prediction_logs.jsonl` — local audit fallback when Postgres is offline.

## Training / serving contract

- Champion is selected by `val_hpo` PR-AUC, never by test metrics; test is scored once.
- Calibration: `CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")` on `val_cal`. Never use `cv="prefit"`.
- Conformal: split nonconformity at `alpha=0.10`; sets are `{0}`, `{1}` or `{0,1}`. An ambiguous `{0,1}` always sets `requires_manual_analyst_review`.
- Baselines: `Rule_CVSS_Only`, `Rule_EPSS_Only`, `LogisticRegression` (median impute → scale → LR), `RandomForest` (`balanced_subsample`).
- MLflow experiment name is `AegisKEV_180d_Exploit_Prediction`; the 3-stage LightGBM ablation is logged as nested runs.
- Frontend metrics come only from `GET /api/v1/metrics/benchmark`; never hardcode benchmark numbers. Missing metrics show the "run train_pipeline.py" state.

## Gotchas

- Ingestion caches NVD yearly `.json.gz` and EPSS daily archives; `--force` re-downloads.
- EPSS archives are gzip both from HTTP and from `data/raw/epss_cache/**/*.csv.gz`; always pass bytes through `decode_gzip_text` (nested-gzip loop) before UTF-8 decoding.
- `nvd_parsed.parquet` must expose `cwe_primary`, `description_en`, `primary_vendor`, the CVSS/vector/CPE/reference fields listed in `sync_data.NVD_PARSED_COLUMNS`; `build_dataset.REQUIRED_NVD_COLUMNS` validates them and fails loudly on drift.
- `add_domain_interaction_features` expects `anchor_date` for `epss_same_day_pct`; missing optional columns become NaN/False rather than raising.
- Keep `effective_anchor_date` through every join — both the target and the prior depend on it.
- `StackedEnsemble` must stay a valid sklearn classifier (`ClassifierMixin, BaseEstimator`, fitted attrs with trailing `_`, `classes_`) or `CalibratedClassifierCV(FrozenEstimator(...))` fails validation.
- The API falls back Postgres → SQLite → JSONL; never let an audit-log failure turn a successful prediction into a 5xx.
- `frontend` uses Next.js `output: "standalone"`, so the Docker runtime image serves `server.js` and copies `.next/static` separately.
