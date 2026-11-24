<div align="center">

# AegisKEV

**Point-in-Time 180-Day CISA KEV Exploitation Forecasting & Calibrated Triage Workbench**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![MLflow 3.18](https://img.shields.io/badge/MLflow-3.18-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Anti--Leakage_Tests-69_Passing-2EA44F?style=for-the-badge&logo=pytest&logoColor=white)](#verification--anti-leakage-test-suite)

*Predicting real-world vulnerability weaponization at $t_0 + 1\text{ day}$ post-disclosure across 257,183 NIST NVD CVEs with calibrated probabilities, split conformal uncertainty sets, and exact TreeSHAP explanations.*

[Live Demo](#quick-start) · [System Architecture](#system-architecture) · [Verified Benchmarks](#verified-holdout-benchmarks-n--63161) · [Anti-Leakage Engineering](#anti-leakage-engineering-guardrails)

</div>

---

## Executive Summary

Enterprise security teams face more than 40,000 newly disclosed vulnerabilities each year, yet fewer than **0.3%** are weaponized in the wild within 180 days. Existing prioritization standards leave critical operational blind spots:

* **Static Severity Scores (CVSS):** Over-alert on theoretical impact (`0.0043` holdout PR-AUC, `0.240` Expected Calibration Error), drowning triage queues in false positives.
* **CISA Known Exploited Vulnerabilities (KEV):** Authoritative ground truth of active exploitation, but strictly reactive—entries are published *after* victims are already targeted.
* **FIRST EPSS:** Models a rolling 30-day exploitation window rather than a 180-day enterprise patch cycle, undergoes structural distribution shifts across model version updates (e.g., EPSS v3 to v4 in March 2025), and provides no per-CVE feature attribution.

**AegisKEV** reconstructs exact point-in-time snapshots 24 hours after CVE publication ($t_0 + 1\text{d}$) and forecasts 180-day CISA KEV inclusion without temporal leakage. On **63,161 untouched post-March 2025 holdout CVEs**, our Optuna-tuned **LightGBM** model achieves **4.1x higher PR-AUC (`0.0370` vs. `0.0090`, $p = 0.003$)** and **5.2x higher Recall@Top-1% (`23.0%` vs. `4.4%`)** over standalone EPSS.

| Corpus Scale | Holdout Evaluation Set | Best Holdout PR-AUC | Top-1% Triage Capture | Calibration Error (ECE) |
| :--- | :--- | :--- | :--- | :--- |
| **257,183** NVD CVEs | **63,161** Unseen CVEs | **0.0370** (**4.1x** vs. EPSS) | **23.0%** (**5.2x** vs. EPSS) | **0.0026** (Platt Calibrated) |

---

## Interface & Workbench Previews

### 1. Live CVE Risk Scorer & Local TreeSHAP Attribution
Interactive single-CVE scoring with real-world vulnerability presets (`Log4Shell`, `MOVEit Transfer`, `ProxyShell`, `Ivanti Connect Secure`), calibrated 180-day probability, split conformal prediction set, and local TreeSHAP waterfall drivers.

![AegisKEV Risk Scorer Workbench](docs/screenshots/risk-scorer.png)

### 2. Verified Holdout Benchmarks & Feature Ablation Explorer
Interactive comparison across 8 evaluated models and rule baselines, paired bootstrap confidence intervals, 3-stage ablation progression, and global TreeSHAP feature importance.

![AegisKEV Benchmark Explorer](docs/screenshots/benchmark.png)

### 3. Unmatured CVE Live Triage Queue
Real-time scoring queue over **76,346 right-censored recent CVEs** awaiting 180-day maturation, flagging high-priority threats and uncertain `{0, 1}` cases for analyst review.

![AegisKEV Live Triage Queue](docs/screenshots/triage-queue.png)

---

## System Architecture

### End-to-End ML & Serving Pipeline

```mermaid
flowchart LR
    subgraph Sources["1. External Threat Feeds (2021 - 2026)"]
        NVD["NIST NVD 2.0 Feeds<br/>(257,183 Valid CVEs)"]
        KEV["CISA KEV Catalog<br/>(1,792 Exploited CVEs)"]
        EPSS["FIRST EPSS Daily Archives<br/>(242,290 Point-in-Time Rows)"]
    end

    subgraph Temporal["2. Point-in-Time Alignment (t0 + 1d)"]
        STRIP["Strip Future cisa* Fields<br/>& Enqueue t_anchor = t0 + 1d"]
        CENSOR["180-Day Right-Censoring<br/>(Cutoff: 2026-05-27)"]
        SPLIT["Chronological Split<br/>Train / Val-HPO / Val-Cal / Test"]
    end

    subgraph Features["3. Leak-Free Feature Engineering (49 Signals)"]
        BASE["Surface & CVSS Vectors<br/>(19 Base Signals)"]
        CAUSAL["Causal CWE & Vendor Priors<br/>(180d Maturation Queue)"]
        RACE["Same-Day EPSS Cohort Rank<br/>& Exploit-vs-Patch Race"]
        TEXT["English Description<br/>TF-IDF + TruncatedSVD (24d)"]
    end

    subgraph Modeling["4. Training, Calibration & UQ"]
        HPO["Optuna Bayesian HPO<br/>(LightGBM / XGBoost / CatBoost)"]
        CAL["FrozenEstimator Platt<br/>Probability Calibration"]
        CONF["Split Conformal Prediction<br/>(alpha = 0.10, q_hat = 0.0039)"]
        SHAP["Exact TreeExplainer<br/>Global & Local SHAP"]
    end

    subgraph Serving["5. Production Stack"]
        API["FastAPI Inference Service<br/>(/predict, /benchmark, /triage)"]
        DB["PostgreSQL / SQLite<br/>Prediction Audit Trail"]
        UI["Next.js 15 SOC Workbench<br/>(Recharts + Tailwind CSS)"]
    end

    NVD & KEV & EPSS --> STRIP --> CENSOR --> SPLIT
    SPLIT --> BASE & CAUSAL & RACE & TEXT
    BASE & CAUSAL & RACE & TEXT --> HPO --> CAL --> CONF --> SHAP
    SHAP --> API --> DB
    API --> UI

    style Sources fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style Temporal fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    style Features fill:#0f172a,stroke:#34d399,stroke-width:2px,color:#f8fafc
    style Modeling fill:#0f172a,stroke:#f472b6,stroke-width:2px,color:#f8fafc
    style Serving fill:#0f172a,stroke:#fbbf24,stroke-width:2px,color:#f8fafc