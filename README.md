<div align="center">

<img src="docs/charts/banner.png" alt="AegisKEV" width="100%"/>

# AegisKEV

### Point-in-Time 180-Day CISA KEV Exploitation Forecasting and Calibrated Triage Workbench

*Freeze the clock at disclosure. Predict exploitation. Quantify uncertainty. Explain every score.*

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-brightgreen?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.18-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-69%20Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

<br/>

![CI](https://img.shields.io/github/actions/workflow/status/vishweshwaranbalasundaram/AegisKEV/ci.yml?style=for-the-badge&label=CI)

<br/>

[Project Dashboard](#project-dashboard) | [Architecture](#system-architecture) | [Guardrails](#seven-anti-leakage-guardrails) | [Benchmark](#verified-holdout-benchmark) | [Screenshots](#analyst-workbench-screenshots) | [Quick Start](#quick-start)

</div>

---

## Table of Contents

1. [Project Dashboard](#project-dashboard)
2. [Problem Statement](#problem-statement)
3. [Solution](#solution)
4. [Data Corpus and Chronological Splits](#data-corpus-and-chronological-splits)
5. [Seven Anti-Leakage Guardrails](#seven-anti-leakage-guardrails)
6. [Feature Engineering](#feature-engineering)
7. [Modeling, Calibration, Conformal UQ and Explainability](#modeling-calibration-conformal-uq-and-explainability)
8. [Verified Holdout Benchmark](#verified-holdout-benchmark)
9. [System Architecture](#system-architecture)
10. [Analyst Workbench Screenshots](#analyst-workbench-screenshots)
11. [API Surface](#api-surface)
12. [Quick Start](#quick-start)
13. [Repository Layout](#repository-layout)
14. [Author](#author)

---

## Project Dashboard

A single-glance view of the verified numbers behind AegisKEV.

<img src="docs/charts/kpi_dashboard.png" alt="KPI dashboard" width="100%"/>

| Corpus | Holdout Result | Engineering |
|:---|:---|:---|
| **257,183** non-rejected NVD CVEs | **0.0370** PR-AUC (4.1x over EPSS-only) | **7** anti-leakage guardrails |
| **1,792** CISA KEV advisories | **23.0%** Recall@Top-1% (5.2x over EPSS-only) | **69** passing Pytest tests |
| **242,290** point-in-time EPSS rows | **0.863** ROC-AUC | **49** engineered signals, 3 stages |
| **177,088** matured CVEs | **0.0026** ECE after Platt calibration | **4** chronological partitions |
| **76,346** right-censored CVEs (live queue) | **p = 0.003** vs EPSS (95% CI +0.0092 to +0.0566) | **3** services plus MLflow in Compose |

### Corpus Composition

<img src="docs/charts/corpus.png" alt="Matured vs right-censored" width="100%"/>

### Chronological Partitions

<img src="docs/charts/data_split.png" alt="Partitions" width="100%"/>

---

## Problem Statement

Security teams receive more than 40,000 new CVEs every year, yet fewer than 0.3% are weaponized in the wild within 180 days. Existing signals each fail in a different way:

| Signal | Failure Mode |
|:---|:---|
| **CVSS** | Over-alerts on theoretical severity. On the holdout it scores 0.0043 PR-AUC and 0.240 ECE. |
| **CISA KEV** | Authoritative ground truth, but reactive: a CVE is added only after active exploitation. |
| **FIRST EPSS** | Models a 30-day window, shifts across model versions (notably EPSS v3 to v4 in March 2025), and offers no per-CVE feature attribution. |

## Solution

AegisKEV freezes the observation clock at:

```
t_anchor = published_date + 1 day   (UTC midnight)
```

and predicts whether a newly disclosed CVE will be added to CISA KEV within 180 days. Every score is paired with:

- a **calibrated probability** (Platt sigmoid on a held-out calibration window),
- a **split conformal uncertainty set** (`{0}`, `{1}`, or `{0, 1}`) that routes ambiguous cases to a human analyst,
- **exact TreeSHAP** global and per-CVE local explanations.

---

## Data Corpus and Chronological Splits

### Raw Ingestion Sources

| Source | Volume | Notes |
|:---|---:|:---|
| NIST NVD 2.0 JSON feeds (2021-2026) | 257,183 non-rejected CVEs | Nested-gzip safe decoding |
| CISA KEV Catalog | 1,792 advisories | Ground-truth label source |
| FIRST EPSS Historical Daily Archives | 242,290 anchored rows | Joined on `t_anchor` |

### Maturation and Right-Censoring (Snapshot Cutoff: 2026-05-27)

| Cohort | Count | Treatment |
|:---|---:|:---|
| Matured CVEs (full 180-day horizon observed) | 177,088 | Used for training and evaluation |
| Right-censored CVEs (disclosed after 2026-05-27) | 76,346 | Excluded from training and evaluation to prevent false-negative label corruption; served in the Live Triage Queue |

### Chronological Partitions

The partitions deliberately straddle the March 2025 EPSS v4 regime shift to test generalization.

| Partition | Date Range | Rows | Positive Rate | Purpose |
|:---|:---|---:|---:|:---|
| Train | 2021-01-01 to 2024-06-30 | 84,966 | 0.301% | Model fitting |
| Val HPO | 2024-07-01 to 2024-11-15 | 15,919 | 0.239% | Optuna hyperparameter tuning |
| Val Calibration | 2024-11-16 to 2025-03-16 | 13,042 | 0.284% | Platt calibration and conformal quantile |
| Holdout Test | 2025-03-17 to 2026-05-27 | 63,161 | 0.214% | Untouched final evaluation (135 positive KEV events) |

---

## Seven Anti-Leakage Guardrails

All seven guardrails are verified by **69 Pytest tests**.

| # | Guardrail | What It Enforces |
|:-:|:---|:---|
| 1 | **Future Field Stripping** | Drops NVD `cisaExploitAdd`, `cisaActionDue`, `cisaRequiredAction`, and `cisaVulnerabilityName`, which are appended post-exploitation. |
| 2 | **Exact Anchor Clock** | Sets `t_anchor = published_date + 1 day`. |
| 3 | **Temporal Eligibility** | Excludes CVEs already in CISA KEV on or before `t_anchor`. |
| 4 | **180-Day Right-Censoring** | Excludes CVEs with incomplete 180-day windows from training and evaluation. |
| 5 | **Point-in-Time EPSS Lookup** | Joins strictly against the historical FIRST EPSS daily CSV published on `t_anchor`. |
| 6 | **Delayed Causal Group Priors** | Smoothed Bayesian m-estimate exploitation rates for `cwe_primary` and `primary_vendor`, using a strict 180-day maturation queue (outcomes revealed only after `published_date + 180d <= current t_anchor`). |
| 7 | **Scale-Invariant Cohort Normalization** | Computes `epss_same_day_pct` (within-day percentile rank) and `epss_minus_cvss_norm` so features remain stable across the March 2025 EPSS v4 probability rescaling. |

### Leakage-Safe Timeline for a Single CVE

```mermaid
%%{init: {'theme':'dark'}}%%
timeline
    title Information Available at t_anchor
    section Known at anchor
        Disclosure day : NVD record published : CVSS and CWE and CPE and references
        Anchor (+1 day) : EPSS daily CSV for t_anchor : Delayed group priors from matured CVEs only
    section Hidden until later
        Post-exploitation : cisaExploitAdd and related fields stripped
        Up to +180 days : KEV outcome label revealed to the priors queue only after maturation
```

### Guardrail Pipeline

```mermaid
%%{init: {'theme':'dark','flowchart':{'curve':'basis'}}}%%
flowchart LR
    A[Raw NVD record]:::raw --> G1[G1 Strip future fields]:::guard
    G1 --> G2[G2 Set t_anchor]:::guard
    G2 --> G3{G3 Already in KEV at or before anchor?}:::decision
    G3 -- yes --> X[Dropped]:::drop
    G3 -- no --> G4{G4 Full 180-day window observed?}:::decision
    G4 -- no --> Q[Live Triage Queue]:::queue
    G4 -- yes --> G5[G5 Point-in-time EPSS join]:::guard
    G5 --> G6[G6 Delayed causal priors]:::guard
    G6 --> G7[G7 Scale-invariant cohort features]:::guard
    G7 --> T[Training and evaluation set]:::ok

    classDef raw fill:#1f2937,stroke:#94a3b8,color:#e2e8f0
    classDef guard fill:#0e7490,stroke:#22d3ee,color:#ecfeff
    classDef decision fill:#7c2d12,stroke:#fb923c,color:#fff7ed
    classDef drop fill:#7f1d1d,stroke:#f87171,color:#fef2f2
    classDef queue fill:#581c87,stroke:#c084fc,color:#faf5ff
    classDef ok fill:#14532d,stroke:#4ade80,color:#f0fdf4
```

---

## Feature Engineering

49 signals are built across three cumulative stages.

| Stage | Feature Count | Contents |
|:---|:-:|:---|
| **Stage 1: Base NVD + EPSS** | 19 | CVSS base, exploitability and impact scores; attack vector, complexity, privileges, user interaction and scope flags; CWE count; CPE config and vendor counts; reference count; vendor advisory, patch and exploit tag flags; anchored EPSS, EPSS percentile, EPSS logit, missing flag. |
| **Stage 2: Causal Priors and Exploit-vs-Patch Race** | 25 | Adds `cwe_kev_prior`, `primary_vendor_kev_prior`, `epss_same_day_pct`, `epss_minus_cvss_norm`, `exploit_without_patch_race`, `zero_click_network_flaw`, `exploit_to_impact_ratio`. |
| **Stage 3: Text Semantics** | 49 | Adds 24-component TF-IDF plus TruncatedSVD embeddings over English CVE descriptions. |

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart TB
    subgraph S1[Stage 1 - 19 features]
        direction LR
        a1[CVSS scores and vector flags]:::s1
        a2[CWE and CPE and reference counts]:::s1
        a3[Advisory patch exploit tags]:::s1
        a4[Anchored EPSS and percentile and logit]:::s1
    end
    subgraph S2[Stage 2 - 25 features]
        direction LR
        b1[cwe_kev_prior]:::s2
        b2[primary_vendor_kev_prior]:::s2
        b3[epss_same_day_pct]:::s2
        b4[epss_minus_cvss_norm]:::s2
        b5[exploit_without_patch_race]:::s2
        b6[zero_click_network_flaw]:::s2
        b7[exploit_to_impact_ratio]:::s2
    end
    subgraph S3[Stage 3 - 49 features]
        direction LR
        c1[TF-IDF over CVE descriptions]:::s3 --> c2[TruncatedSVD 24 components]:::s3
    end
    S1 --> S2 --> S3

    classDef s1 fill:#1e3a8a,stroke:#60a5fa,color:#eff6ff
    classDef s2 fill:#065f46,stroke:#34d399,color:#ecfdf5
    classDef s3 fill:#6b21a8,stroke:#d8b4fe,color:#faf5ff
```

---

## Modeling, Calibration, Conformal UQ and Explainability

### Models Evaluated

| Model | Notes |
|:---|:---|
| LightGBM | Optuna-tuned, 25 trials, val_hpo PR-AUC 0.0841 |
| XGBoost | val_hpo PR-AUC 0.0846 |
| CatBoost | val_hpo PR-AUC 0.0984, selected as the val_hpo champion |
| Stacked Ensemble | 3-fold expanding-window |
| Logistic Regression | Linear baseline |
| Random Forest | Bagged-tree baseline |
| Rule_EPSS_Only | Rule baseline |
| Rule_CVSS_Only | Rule baseline |

### Calibration

Post-hoc Platt sigmoid calibration through `FrozenEstimator`, fitted on `val_cal`.

### Split Conformal Uncertainty Quantification

Configured at alpha = 0.10 with `q_hat = 0.00389`. Output is a prediction set of `{0}`, `{1}`, or `{0, 1}`; ambiguous cases are flagged for human analyst review.

| Conformal Metric | Value |
|:---|---:|
| Empirical coverage across the EPSS v4 regime shift | 81.7% |
| Empty sets | 18.2% |
| Average set size | 0.818 |

<img src="docs/charts/conformal.png" alt="Conformal" width="100%"/>

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart LR
    P[Calibrated probability p]:::p --> C{Compare against q_hat 0.00389}:::d
    C --> S0["Set {0} - confident benign"]:::green
    C --> S1["Set {1} - confident exploit candidate"]:::red
    C --> SB["Set {0, 1} - uncertain, route to analyst"]:::amber
    C --> SE["Empty set - abstain"]:::grey

    classDef p fill:#0e7490,stroke:#22d3ee,color:#ecfeff
    classDef d fill:#312e81,stroke:#818cf8,color:#eef2ff
    classDef green fill:#14532d,stroke:#4ade80,color:#f0fdf4
    classDef red fill:#7f1d1d,stroke:#f87171,color:#fef2f2
    classDef amber fill:#78350f,stroke:#fbbf24,color:#fffbeb
    classDef grey fill:#1f2937,stroke:#9ca3af,color:#f3f4f6
```

### Explainability

Exact `shap.TreeExplainer` provides global feature importances and local per-CVE waterfall contributions.

---

## Verified Holdout Benchmark

All figures are computed on **n = 63,161 untouched test CVEs** (135 positive KEV events).

| Model | PR-AUC | ROC-AUC | Recall@Top-1% | Precision@1% | RVC@500 | ECE | p-value vs EPSS |
|:---|---:|---:|:---|---:|---:|---:|---:|
| **LightGBM (Optuna Tuned)** | **0.0370** | 0.863 | **23.0%** (31/135) | 4.91% | 0.299 | 0.0026 | 0.003 |
| XGBoost (Optuna Tuned) | 0.0279 | 0.871 | 21.5% (29/135) | 4.59% | 0.281 | 0.0027 | 0.009 |
| CatBoost (Val-HPO Champion) | 0.0225 | 0.862 | 19.3% (26/135) | 4.11% | 0.365 | 0.0036 | 0.024 |
| Stacked Ensemble | 0.0178 | 0.838 | 15.6% (21/135) | n/a | 0.306 | 0.0024 | 0.058 |
| Logistic Regression | 0.0160 | 0.847 | 13.3% (18/135) | n/a | 0.178 | 0.0057 | 0.067 |
| Random Forest | 0.0156 | 0.769 | 11.9% (16/135) | n/a | 0.137 | 0.0028 | 0.086 |
| Rule_EPSS_Only | 0.0090 | 0.741 | 4.4% (6/135) | 0.95% | 0.031 | 0.0016 | baseline |
| Rule_CVSS_Only | 0.0043 | 0.606 | 5.2% (7/135) | 1.11% | 0.124 | 0.2402 | n/a |

**LightGBM headline results**

- PR-AUC is 4.1x the EPSS-only rule; Recall@Top-1% is 5.2x; RVC@500 is 9.6x.
- F1@1% is 0.0808; Brier score is 0.0022.
- Delta PR-AUC vs EPSS: +0.0288, 95% CI [+0.0092, +0.0566], p = 0.003.

**Paired PR-AUC deltas vs EPSS-only (95% bootstrap CI)**

| Model | Delta PR-AUC | 95% CI | p |
|:---|---:|:---|---:|
| LightGBM | +0.0288 | [+0.0092, +0.0566] | 0.003 |
| XGBoost | +0.0207 | [+0.0048, +0.0422] | 0.009 |
| CatBoost | +0.0135 | [+0.0004, +0.0251] | 0.024 |

Note: Rule_CVSS_Only has a Brier score of 0.1827 and an ECE of 0.2402, which is the over-alerting failure mode described in the problem statement.

### PR-AUC on the Holdout

<img src="docs/charts/benchmark_prauc.png" alt="PR-AUC" width="100%"/>

### Recall@Top-1% on the Holdout

<img src="docs/charts/recall_top1.png" alt="Recall top 1%" width="100%"/>

### Model Fingerprint

<img src="docs/charts/radar.png" alt="Radar" width="100%"/>

### Scorecard Heatmap

<img src="docs/charts/scorecard_heatmap.png" alt="Heatmap" width="100%"/>

### Calibration Quality

<img src="docs/charts/calibration_ece.png" alt="ECE" width="100%"/>

### Statistical Significance

<img src="docs/charts/delta_ci_forest.png" alt="Forest plot" width="100%"/>

### Top-1% Triage Capture (ASCII)

```
Share of the 135 holdout KEV events captured inside the top 1% of the ranked queue

LightGBM (Tuned)  | ############################################## 23.0%  (31/135)
XGBoost (Tuned)   | ###########################################    21.5%  (29/135)
CatBoost          | #######################################        19.3%  (26/135)
Stacked Ensemble  | ###############################                15.6%  (21/135)
Logistic Regr.    | ###########################                    13.3%  (18/135)
Random Forest     | ########################                       11.9%  (16/135)
Rule_CVSS_Only    | ##########                                      5.2%   (7/135)
Rule_EPSS_Only    | #########                                       4.4%   (6/135)
                  +----------------------------------------------
                    each # = 0.5 percentage points
```

### PR-AUC Histogram (ASCII)

```
Holdout PR-AUC (each # = 0.001)

LightGBM  | ##################################### 0.0370
XGBoost   | ############################          0.0279
CatBoost  | ######################                0.0225
Stacked   | ##################                    0.0178
LogReg    | ################                      0.0160
RandForest| ################                      0.0156
EPSS-only | #########                             0.0090
CVSS-only | ####                                  0.0043
```

### Three-Stage Ablation

| Stage | Features | PR-AUC | ROC-AUC | Recall@1% | RVC@500 | ECE |
|:---|:-:|---:|---:|---:|---:|---:|
| Stage 1: Base NVD + EPSS | 19 | 0.0136 | 0.825 | 8.1% | 0.034 | 0.0028 |
| Stage 2: + Causal Priors and Race | 25 | **0.0212** (+56% relative) | 0.825 | **16.3%** (2.0x) | **0.286** (8.5x) | 0.0021 |
| Stage 3: + Text SVD Embeddings | 49 | 0.0166 | **0.854** | 9.6% | 0.119 | 0.0030 |

<img src="docs/charts/ablation.png" alt="Ablation" width="100%"/>

### Global TreeSHAP Importance (ASCII)

```
Mean absolute SHAP value (each # = about 0.033)

reference_count          | ######################################## 1.333
tfidf_svd_03             | ##################                       0.584
has_vendor_advisory_ref  | ########                                 0.257
cvss_base_score          | ######                                   0.193
tfidf_svd_20             | ####                                     0.120
epss_same_day_pct        | #                                        0.040
cvss_impact_score        | #                                        0.030
cpe_config_count         | #                                        0.024
exploit_to_impact_ratio  | #                                        0.017
```

<img src="docs/charts/shap_global.png" alt="SHAP" width="100%"/>

---

## System Architecture

### 1. End-to-End Ingestion, Anti-Leakage Feature Engineering, Training and Serving

```mermaid
%%{init: {'theme':'dark','flowchart':{'curve':'basis'}}}%%
flowchart TB
    subgraph SRC[Data Sources]
        direction LR
        NVD[(NIST NVD 2.0 JSON feeds 2021-2026)]:::src
        KEV[(CISA KEV Catalog)]:::src
        EPSS[(FIRST EPSS Daily Archives)]:::src
    end

    subgraph ING[Ingestion and Anti-Leakage Engineering]
        direction TB
        P1[Nested-gzip safe NVD parser]:::ing
        P2[Future field stripping]:::ing
        P3[Anchor clock t_anchor = published + 1d]:::ing
        P4[Temporal eligibility and 180d right-censoring]:::ing
        P5[Point-in-time EPSS join]:::ing
        P6[Delayed causal group priors]:::ing
        P7[Scale-invariant cohort features]:::ing
        P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7
    end

    subgraph FE[Feature Store - 49 signals]
        direction LR
        F1[Stage 1 Base 19]:::fe
        F2[Stage 2 Priors and Race 25]:::fe
        F3[Stage 3 Text SVD 49]:::fe
        F1 --> F2 --> F3
    end

    subgraph TRN[Training and Calibration]
        direction TB
        T1[Optuna HPO on val_hpo]:::trn
        T2[LightGBM / XGBoost / CatBoost / Stack / LR / RF]:::trn
        T3[Platt via FrozenEstimator on val_cal]:::trn
        T4[Split conformal q_hat on val_cal]:::trn
        T5[TreeSHAP explainer]:::trn
        T1 --> T2 --> T3 --> T4 --> T5
    end

    subgraph TRK[Experiment Tracking]
        ML[(MLflow 3.18)]:::trk
    end

    subgraph SRV[Serving Layer]
        direction TB
        API[FastAPI plus Pydantic v2]:::srv
        AUD[(PostgreSQL / SQLite / JSONL audit log)]:::srv
        API --> AUD
    end

    subgraph UI[SOC Analyst Workbench]
        WEB[Next.js 15 App Router]:::ui
    end

    NVD --> P1
    KEV --> P4
    EPSS --> P5
    P7 --> F1
    F3 --> T1
    T2 -. runs and metrics .-> ML
    T5 --> API
    API <--> WEB

    classDef src fill:#1e293b,stroke:#94a3b8,color:#e2e8f0
    classDef ing fill:#0e7490,stroke:#22d3ee,color:#ecfeff
    classDef fe fill:#065f46,stroke:#34d399,color:#ecfdf5
    classDef trn fill:#6b21a8,stroke:#d8b4fe,color:#faf5ff
    classDef trk fill:#1d4ed8,stroke:#93c5fd,color:#eff6ff
    classDef srv fill:#9a3412,stroke:#fdba74,color:#fff7ed
    classDef ui fill:#be185d,stroke:#f9a8d4,color:#fdf2f8
```

### 2. Runtime Inference and Conformal UQ Sequence

```mermaid
%%{init: {'theme':'dark'}}%%
sequenceDiagram
    autonumber
    actor Analyst as SOC Analyst
    participant UI as Next.js Workbench
    participant API as FastAPI
    participant FE as Feature Builder
    participant M as Calibrated Model
    participant CP as Conformal Layer
    participant SH as TreeSHAP
    participant DB as Audit Log

    Analyst->>UI: Submit CVE for scoring
    UI->>API: POST /api/v1/predict
    API->>FE: Build 49-feature vector at t_anchor
    FE-->>API: Features (point-in-time safe)
    API->>M: Score features
    M-->>API: Platt-calibrated probability
    API->>CP: Compare against q_hat 0.00389
    CP-->>API: Prediction set {0}, {1} or {0, 1}
    API->>SH: Explain prediction
    SH-->>API: Local waterfall contributions
    API->>DB: Persist prediction audit record
    API-->>UI: Probability, set, SHAP waterfall
    UI-->>Analyst: Risk card plus waterfall plus review flag
    opt Set is {0, 1}
        UI-->>Analyst: Flag for human analyst review
    end
```

### 3. Container Topology

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart LR
    DEV([Developer / Analyst]):::person
    subgraph DC[docker-compose.yml]
        direction LR
        FE[Frontend<br/>Next.js 15]:::fe
        BE[API<br/>FastAPI]:::be
        PG[(PostgreSQL)]:::db
        ML[MLflow<br/>3.18]:::ml
    end
    DEV --> FE
    FE -->|REST| BE
    BE -->|audit log| PG
    BE -. model lineage .-> ML

    classDef person fill:#334155,stroke:#cbd5e1,color:#f8fafc
    classDef fe fill:#be185d,stroke:#f9a8d4,color:#fdf2f8
    classDef be fill:#0f766e,stroke:#5eead4,color:#f0fdfa
    classDef db fill:#1d4ed8,stroke:#93c5fd,color:#eff6ff
    classDef ml fill:#7c3aed,stroke:#c4b5fd,color:#f5f3ff
```

### 4. CI and Delivery Pipeline

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart LR
    C[git push / PR]:::a --> W[GitHub Actions<br/>.github/workflows/ci.yml]:::b
    W --> T[Pytest suite<br/>69 anti-leakage and API tests]:::c
    T --> B[Multi-stage Docker build]:::d
    B --> R[Compose-ready images]:::e

    classDef a fill:#334155,stroke:#cbd5e1,color:#f8fafc
    classDef b fill:#1d4ed8,stroke:#93c5fd,color:#eff6ff
    classDef c fill:#15803d,stroke:#86efac,color:#f0fdf4
    classDef d fill:#b45309,stroke:#fcd34d,color:#fffbeb
    classDef e fill:#7e22ce,stroke:#d8b4fe,color:#faf5ff
```

### 5. Analyst Workbench Component Map

```mermaid
%%{init: {'theme':'dark'}}%%
flowchart TB
    APP[Next.js 15 App Router]:::root
    APP --> K[Executive KPI Strip]:::c1
    APP --> RS[Risk Scorer]:::c2
    APP --> BM[Benchmark and Ablation Charts - Recharts]:::c3
    APP --> TQ[Live Triage Queue over unmatured CVEs]:::c4
    APP --> RP[One-click Markdown and JSON reports]:::c5
    RS --> G[Global TreeSHAP chart]:::c6
    RS --> L[Local TreeSHAP waterfall]:::c6
    RS --> U[Conformal prediction set badge]:::c6

    classDef root fill:#0f172a,stroke:#38bdf8,color:#e0f2fe
    classDef c1 fill:#0e7490,stroke:#22d3ee,color:#ecfeff
    classDef c2 fill:#be185d,stroke:#f9a8d4,color:#fdf2f8
    classDef c3 fill:#6d28d9,stroke:#c4b5fd,color:#f5f3ff
    classDef c4 fill:#b45309,stroke:#fcd34d,color:#fffbeb
    classDef c5 fill:#15803d,stroke:#86efac,color:#f0fdf4
    classDef c6 fill:#9f1239,stroke:#fda4af,color:#fff1f2
```

---

## Analyst Workbench Screenshots

Next.js 15 SOC Analyst Workbench with a dark cyber-intelligence UI.

### Risk Scorer

Calibrated probability, conformal prediction set, and local TreeSHAP waterfall for a single CVE.

![Risk Scorer](docs/screenshots/risk-scorer.png)

### Benchmark and Ablation

Holdout benchmark comparison and the three-stage ablation charts.

![Benchmark](docs/screenshots/benchmark.png)

### Live Triage Queue

Right-censored (unmatured) CVEs ranked for analyst review.

![Triage Queue](docs/screenshots/triage-queue.png)

---

## API Surface

Backend: FastAPI (`backend/app/main.py`) with Pydantic v2 schemas (`schemas.py`) and PostgreSQL / SQLite / JSONL prediction audit logging.

| Method | Endpoint | Purpose |
|:---|:---|:---|
| GET | `/health` | Service health check |
| POST | `/api/v1/predict` | Calibrated score, conformal set, and TreeSHAP explanation for a CVE |
| GET | `/api/v1/metrics/benchmark` | Verified holdout benchmark and ablation metrics |
| GET | `/api/v1/triage/recent` | Live Triage Queue over unmatured CVEs |
| GET | `/api/v1/reports/markdown` | Downloadable analyst report |

---

## Quick Start

```bash
git clone https://github.com/vishweshwaranbalasundaram/AegisKEV.git
cd AegisKEV

# Full stack: API, Frontend, PostgreSQL, MLflow
docker compose up --build
```

Regenerate the README charts:

```bash
python scripts/generate_charts.py
```

Run the anti-leakage test suite locally:

```bash
pytest
```

Service ports and environment variables are defined in `docker-compose.yml`.

---

## Repository Layout

```
AegisKEV/
|-- backend/
|   `-- app/
|       |-- main.py            FastAPI application
|       `-- schemas.py         Pydantic v2 schemas
|-- frontend/                  Next.js 15 App Router SOC Analyst Workbench
|-- docs/
|   |-- charts/                Generated README visuals (PNG)
|   `-- screenshots/
|       |-- risk-scorer.png
|       |-- benchmark.png
|       `-- triage-queue.png
|-- scripts/
|   `-- generate_charts.py     Regenerates docs/charts from verified metrics
|-- .github/
|   `-- workflows/ci.yml       GitHub Actions CI
|-- Dockerfile                 Multi-stage build
|-- docker-compose.yml         API, Frontend, PostgreSQL, MLflow
`-- README.md
```

---

## Author

**Vishweshwaran** - Computer Science and Engineering, Rajalakshmi Engineering College, Chennai.

GitHub: [vishweshwaranbalasundaram](https://github.com/vishweshwaranbalasundaram)
