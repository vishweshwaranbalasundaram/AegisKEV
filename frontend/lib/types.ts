export type RiskTier = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
export type ConformalSet = "{0}" | "{1}" | "{0,1}";

export interface ShapDriver {
  feature: string;
  contribution: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface Prediction {
  cve_id: string;
  probability: number;
  risk_tier: RiskTier;
  conformal_set: ConformalSet;
  requires_manual_analyst_review: boolean;
  top_drivers: ShapDriver[];
  shap_base_value?: number | null;
  shap_values?: ShapDriver[];
  model_name: string;
  caveat: string;
  scored_at: string;
}

export interface PredictPayload {
  cve_id: string;
  cvss_base_score: number;
  cvss_exploitability_score: number;
  cvss_impact_score: number;
  is_network_av: boolean;
  is_low_ac: boolean;
  is_priv_none: boolean;
  is_ui_none: boolean;
  scope_changed: boolean;
  cwe_count: number;
  cpe_config_count: number;
  cpe_vendor_count: number;
  reference_count: number;
  has_vendor_advisory_ref: boolean;
  has_patch_tag: boolean;
  has_exploit_tag: boolean;
  epss: number;
  epss_percentile: number;
  cwe_kev_prior: number;
  primary_vendor_kev_prior: number;
}

export type PredictionRequest = Omit<
  PredictPayload,
  | "cvss_base_score"
  | "cvss_exploitability_score"
  | "cvss_impact_score"
  | "epss"
  | "epss_percentile"
  | "cwe_kev_prior"
  | "primary_vendor_kev_prior"
> & {
  cvss_base_score: number | null;
  cvss_exploitability_score: number | null;
  cvss_impact_score: number | null;
  epss: number | null;
  epss_percentile: number | null;
  cwe_kev_prior: number | null;
  primary_vendor_kev_prior: number | null;
  epss_logit?: number | null;
  epss_missing_flag?: number | null;
  epss_same_day_pct?: number | null;
  epss_minus_cvss_norm?: number | null;
  exploit_without_patch_race?: number | null;
  zero_click_network_flaw?: number | null;
  exploit_to_impact_ratio?: number | null;
};

export interface ModelMetrics {
  pr_auc?: number;
  roc_auc?: number;
  recall_top_1pct?: number;
  precision_top_1pct?: number;
  f1_top_1pct?: number;
  rvc_100?: number;
  rvc_500?: number;
  brier?: number;
  ece_10?: number;
  n_features?: number;
  n_rows?: number;
  bootstrap_vs_rule_epss?: {
    delta_mean?: number;
    ci_low?: number;
    ci_high?: number;
    p_value_one_sided?: number;
  };
}

export interface TriageItem {
  cve_id: string;
  published_date?: string | null;
  cvss_base_score?: number | null;
  epss?: number | null;
  probability: number;
  risk_tier: RiskTier;
  conformal_set: ConformalSet;
  requires_manual_analyst_review: boolean;
  top_shap_driver?: string | null;
  top_drivers: ShapDriver[];
  features: Record<string, number | boolean | string | null>;
}

export interface RecentTriage {
  total_unmatured: number;
  cutoff_date?: string | null;
  items: TriageItem[];
}

export interface Benchmark {
  experiment?: string;
  generated_at?: string;
  champion?: { name?: string; val_hpo_pr_auc?: number; selection_metric?: string };
  models?: Record<string, ModelMetrics>;
  ablation?: Record<string, ModelMetrics>;
  conformal?: {
    alpha?: number;
    q_hat?: number;
    coverage?: number;
    ambiguous_fraction?: number;
    avg_set_size?: number;
  };
  shap_top_features?: { feature: string; mean_abs_shap: number }[];
  dataset_manifest?: {
    row_counts?: Record<string, number>;
    unmatured_rows?: number;
    date_ranges?: Record<string, { min?: string; max?: string }>;
  };
  dataset?: {
    rows?: Record<string, number>;
    positive_rate?: Record<string, number>;
  };
}

export type BenchmarkResult =
  | { ok: true; data: Benchmark }
  | { ok: false; detail: string };
