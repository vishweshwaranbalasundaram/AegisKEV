"""Pydantic v2 schemas for the AegisKEV inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CVE_PATTERN = r"^CVE-\d{4}-\d{4,19}$"

RiskTier = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
ConformalSet = Literal["{0}", "{1}", "{0,1}"]


class CveFeatures(BaseModel):
    """Raw NVD/EPSS inputs accepted by ``POST /api/v1/predict``.

    Interaction fields are optional: the API derives them from the raw fields
    when omitted, so a caller can send only NVD + EPSS values.
    """

    model_config = ConfigDict(extra="ignore")

    cve_id: str = Field(
        pattern=CVE_PATTERN,
        description="CVE identifier, e.g. CVE-2021-44228",
        examples=["CVE-2021-44228"],
    )

    # CVSS
    cvss_base_score: float | None = Field(default=None, ge=0.0, le=10.0)
    cvss_exploitability_score: float | None = Field(default=None, ge=0.0)
    cvss_impact_score: float | None = Field(default=None, ge=0.0)
    is_network_av: bool = False
    is_low_ac: bool = False
    is_priv_none: bool = False
    is_ui_none: bool = False
    scope_changed: bool = False

    # NVD structure
    cwe_count: int = Field(default=0, ge=0)
    cpe_config_count: int = Field(default=0, ge=0)
    cpe_vendor_count: int = Field(default=0, ge=0)
    reference_count: int = Field(default=0, ge=0)
    has_vendor_advisory_ref: bool = False
    has_patch_tag: bool = False
    has_exploit_tag: bool = False

    # Point-in-time EPSS
    epss: float | None = Field(default=None, ge=0.0, le=1.0)
    epss_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    epss_logit: float | None = None
    epss_missing_flag: int | None = Field(default=None, ge=0, le=1)

    # Causal priors
    cwe_kev_prior: float | None = Field(default=None, ge=0.0, le=1.0)
    primary_vendor_kev_prior: float | None = Field(default=None, ge=0.0, le=1.0)

    # Domain interactions (optional; derived if omitted)
    epss_same_day_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    epss_minus_cvss_norm: float | None = Field(default=None, ge=-1.0, le=1.0)
    exploit_without_patch_race: int | None = Field(default=None, ge=-1, le=1)
    zero_click_network_flaw: int | None = Field(default=None, ge=0, le=1)
    exploit_to_impact_ratio: float | None = Field(default=None, ge=0.0)


class BatchPredictRequest(BaseModel):
    items: list[CveFeatures] = Field(min_length=1, max_length=100)


class ShapDriver(BaseModel):
    feature: str
    contribution: float
    direction: Literal["increases_risk", "decreases_risk"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    cve_id: str
    probability: float = Field(ge=0.0, le=1.0, description="Calibrated 180-day KEV probability")
    risk_tier: RiskTier
    conformal_set: ConformalSet = Field(
        description="90% split-conformal prediction set over {0,1}"
    )
    requires_manual_analyst_review: bool
    top_drivers: list[ShapDriver] = Field(
        default_factory=list, description="Top local TreeSHAP drivers (max 8)"
    )
    shap_base_value: float | None = Field(
        default=None, description="TreeSHAP expected value in the base model output space"
    )
    shap_values: list[ShapDriver] = Field(
        default_factory=list,
        description="All local TreeSHAP feature contributions in the base model output space",
    )
    model_name: str
    caveat: str
    scored_at: str


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class TriageItem(BaseModel):
    cve_id: str
    published_date: str | None = None
    cvss_base_score: float | None = None
    epss: float | None = None
    probability: float
    risk_tier: RiskTier
    conformal_set: ConformalSet
    requires_manual_analyst_review: bool
    top_shap_driver: str | None = None
    top_drivers: list[ShapDriver] = Field(default_factory=list)
    features: CveFeatures


class RecentTriageResponse(BaseModel):
    total_unmatured: int
    cutoff_date: str | None = None
    items: list[TriageItem]


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None
    detail: str | None = None
