"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { predict } from "@/lib/api";
import { PRESETS } from "@/lib/presets";
import type {
  PredictPayload,
  Prediction,
  PredictionRequest,
  RiskTier,
  TriageItem,
} from "@/lib/types";

const TIER_STYLES: Record<RiskTier, string> = {
  CRITICAL: "border-rose-500/50 bg-rose-500/10 text-rose-300",
  HIGH: "border-orange-500/50 bg-orange-500/10 text-orange-300",
  MEDIUM: "border-amber-500/50 bg-amber-500/10 text-amber-300",
  LOW: "border-emerald-500/50 bg-emerald-500/10 text-emerald-300",
};

const FLAGS: { key: BoolKey; label: string }[] = [
  { key: "is_network_av", label: "Network AV" },
  { key: "is_low_ac", label: "Low AC" },
  { key: "is_priv_none", label: "No privileges" },
  { key: "is_ui_none", label: "No user interaction" },
  { key: "scope_changed", label: "Scope changed" },
  { key: "has_exploit_tag", label: "Exploit reference" },
  { key: "has_patch_tag", label: "Patch reference" },
  { key: "has_vendor_advisory_ref", label: "Vendor advisory" },
];

type BoolKey =
  | "is_network_av"
  | "is_low_ac"
  | "is_priv_none"
  | "is_ui_none"
  | "scope_changed"
  | "has_exploit_tag"
  | "has_patch_tag"
  | "has_vendor_advisory_ref";

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-center justify-between text-xs uppercase tracking-wider text-slate-400">
        {label}
        <span className="font-mono text-slate-200">
          {format ? format(value) : value}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-cyan-400"
      />
    </label>
  );
}

function NumberField({
  label,
  value,
  min,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs uppercase tracking-wider text-slate-400">
        {label}
      </span>
      <input
        type="number"
        min={min}
        value={value}
        onChange={(event) => onChange(Math.max(min, Number(event.target.value) || min))}
        className="w-full rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
      />
    </label>
  );
}

function queuedFeaturesToPayload(item: TriageItem): PredictionRequest {
  const features = item.features;
  const number = (key: string): number => {
    const value = features[key];
    return typeof value === "number" && Number.isFinite(value) ? value : 0;
  };
  const nullableNumber = (key: string): number | null => {
    const value = features[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  const flag = (key: string): boolean => features[key] === true;
  return {
    cve_id: item.cve_id,
    cvss_base_score: nullableNumber("cvss_base_score"),
    cvss_exploitability_score: nullableNumber("cvss_exploitability_score"),
    cvss_impact_score: nullableNumber("cvss_impact_score"),
    is_network_av: flag("is_network_av"),
    is_low_ac: flag("is_low_ac"),
    is_priv_none: flag("is_priv_none"),
    is_ui_none: flag("is_ui_none"),
    scope_changed: flag("scope_changed"),
    cwe_count: number("cwe_count"),
    cpe_config_count: number("cpe_config_count"),
    cpe_vendor_count: number("cpe_vendor_count"),
    reference_count: number("reference_count"),
    has_vendor_advisory_ref: flag("has_vendor_advisory_ref"),
    has_patch_tag: flag("has_patch_tag"),
    has_exploit_tag: flag("has_exploit_tag"),
    epss: nullableNumber("epss"),
    epss_percentile: nullableNumber("epss_percentile"),
    cwe_kev_prior: nullableNumber("cwe_kev_prior"),
    primary_vendor_kev_prior: nullableNumber("primary_vendor_kev_prior"),
    epss_logit: nullableNumber("epss_logit"),
    epss_missing_flag: nullableNumber("epss_missing_flag"),
    epss_same_day_pct: nullableNumber("epss_same_day_pct"),
    epss_minus_cvss_norm: nullableNumber("epss_minus_cvss_norm"),
    exploit_without_patch_race: nullableNumber("exploit_without_patch_race"),
    zero_click_network_flaw: nullableNumber("zero_click_network_flaw"),
    exploit_to_impact_ratio: nullableNumber("exploit_to_impact_ratio"),
  };
}

function ShapWaterfall({ prediction }: { prediction: Prediction }) {
  const contributions = prediction.shap_values ?? [];
  const baseValue = prediction.shap_base_value;
  if (typeof baseValue !== "number" || contributions.length === 0) return null;

  let running = baseValue;
  const data = [
    {
      feature: "Base value",
      offset: 0,
      increase: Math.max(baseValue, 0),
      decrease: Math.max(-baseValue, 0),
      contribution: baseValue,
    },
    ...contributions.map((driver) => {
      const before = running;
      running += driver.contribution;
      return {
        feature: driver.feature,
        offset: Math.min(before, running),
        increase: driver.contribution > 0 ? driver.contribution : 0,
        decrease: driver.contribution < 0 ? -driver.contribution : 0,
        contribution: driver.contribution,
      };
    }),
    {
      feature: "Base model output",
      offset: 0,
      increase: Math.max(running, 0),
      decrease: Math.max(-running, 0),
      contribution: running,
    },
  ];

  return (
    <div className="mt-3 max-h-[520px] overflow-y-auto rounded-lg border border-slate-800/80 bg-slate-950/50">
      <div style={{ height: Math.max(280, data.length * 22) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 12, bottom: 8 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
            <YAxis dataKey="feature" type="category" width={175} tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0b1220",
                border: "1px solid rgba(34, 211, 238, 0.25)",
                borderRadius: 8,
                color: "#e2e8f0",
                fontSize: 11,
              }}
              formatter={(value, name, item) =>
                name === "offset"
                  ? null
                  : [
                      Number(item.payload.contribution).toFixed(4),
                      "SHAP contribution",
                    ]
              }
            />
            <Bar dataKey="offset" stackId="waterfall" fill="transparent" legendType="none" isAnimationActive={false} />
            <Bar dataKey="increase" stackId="waterfall" fill="#fb7185" legendType="none" isAnimationActive={false} />
            <Bar dataKey="decrease" stackId="waterfall" fill="#22d3ee" legendType="none" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function TriagePanel({ queuedCve }: { queuedCve: TriageItem | null }) {
  const [form, setForm] = useState<PredictPayload>(PRESETS[0].payload);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!queuedCve) return;
    const payload = queuedFeaturesToPayload(queuedCve);
    setForm({
      ...PRESETS[0].payload,
      ...payload,
      cvss_base_score: payload.cvss_base_score ?? 0,
      cvss_exploitability_score: payload.cvss_exploitability_score ?? 0,
      cvss_impact_score: payload.cvss_impact_score ?? 0,
      epss: payload.epss ?? 0,
      epss_percentile: payload.epss_percentile ?? 0,
      cwe_kev_prior: payload.cwe_kev_prior ?? 0,
      primary_vendor_kev_prior: payload.primary_vendor_kev_prior ?? 0,
    });
    setLoading(true);
    setError(null);
    predict(payload)
      .then(setPrediction)
      .catch((caught: unknown) => {
        setPrediction(null);
        setError(caught instanceof Error ? caught.message : "Prediction failed");
      })
      .finally(() => setLoading(false));
  }, [queuedCve]);

  function update<K extends keyof PredictPayload>(
    key: K,
    value: PredictPayload[K],
  ) {
    setForm((previous) => ({ ...previous, [key]: value }));
  }

  function toggle(key: BoolKey, value: boolean) {
    setForm((previous) => ({ ...previous, [key]: value }));
  }

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setPrediction(await predict(form));
    } catch (caught) {
      setPrediction(null);
      setError(caught instanceof Error ? caught.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  const conformal =
    prediction?.conformal_set === "{1}"
      ? {
          label: "Exploit in 90% conformal set",
          className: "border-rose-500/40 bg-rose-500/10 text-rose-300",
        }
      : prediction?.conformal_set === "{0}"
        ? {
            label: "Non-exploit in 90% conformal set",
            className: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
          }
        : {
            label: "Ambiguous set {0,1}",
            className: "border-amber-500/40 bg-amber-500/10 text-amber-300",
          };

  const maxDriver = Math.max(
    ...(prediction?.top_drivers ?? []).map((driver) =>
      Math.abs(driver.contribution),
    ),
    0.0001,
  );

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      <section className="glass-panel rounded-xl p-5">
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              onClick={() => {
                setForm(preset.payload);
                setPrediction(null);
                setError(null);
              }}
              className={`rounded-full border px-3 py-1 text-xs transition ${
                form.cve_id === preset.payload.cve_id
                  ? "border-cyan-500/60 bg-cyan-500/10 text-cyan-300"
                  : "border-slate-700 text-slate-400 hover:border-slate-500 hover:text-slate-200"
              }`}
              title={preset.note}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <label className="block space-y-1.5 sm:col-span-2">
            <span className="text-xs uppercase tracking-wider text-slate-400">
              CVE identifier
            </span>
            <input
              value={form.cve_id}
              onChange={(event) => update("cve_id", event.target.value)}
              pattern="^CVE-\d{4}-\d{4,19}$"
              className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 font-mono text-sm text-cyan-300 focus:border-cyan-500 focus:outline-none"
            />
          </label>

          <Slider
            label="CVSS base"
            value={form.cvss_base_score}
            min={0}
            max={10}
            step={0.1}
            onChange={(value) => update("cvss_base_score", value)}
            format={(value) => value.toFixed(1)}
          />
          <Slider
            label="EPSS (anchored)"
            value={form.epss}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => update("epss", value)}
            format={(value) => value.toFixed(2)}
          />
          <Slider
            label="CWE KEV prior"
            value={form.cwe_kev_prior}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => update("cwe_kev_prior", value)}
            format={(value) => value.toFixed(2)}
          />
          <Slider
            label="Vendor KEV prior"
            value={form.primary_vendor_kev_prior}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => update("primary_vendor_kev_prior", value)}
            format={(value) => value.toFixed(2)}
          />
          <Slider
            label="EPSS percentile"
            value={form.epss_percentile}
            min={0}
            max={1}
            step={0.01}
            onChange={(value) => update("epss_percentile", value)}
            format={(value) => value.toFixed(2)}
          />
          <Slider
            label="Exploitability score"
            value={form.cvss_exploitability_score}
            min={0}
            max={4}
            step={0.1}
            onChange={(value) => update("cvss_exploitability_score", value)}
            format={(value) => value.toFixed(1)}
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <NumberField
            label="CWE count"
            value={form.cwe_count}
            min={0}
            onChange={(value) => update("cwe_count", value)}
          />
          <NumberField
            label="CPE configs"
            value={form.cpe_config_count}
            min={0}
            onChange={(value) => update("cpe_config_count", value)}
          />
          <NumberField
            label="CPE vendors"
            value={form.cpe_vendor_count}
            min={0}
            onChange={(value) => update("cpe_vendor_count", value)}
          />
          <NumberField
            label="References"
            value={form.reference_count}
            min={0}
            onChange={(value) => update("reference_count", value)}
          />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {FLAGS.map((flag) => (
            <label
              key={flag.key}
              className="flex items-center gap-2 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1.5 text-xs text-slate-300"
            >
              <input
                type="checkbox"
                checked={form[flag.key]}
                onChange={(event) => toggle(flag.key, event.target.checked)}
                className="accent-cyan-500"
              />
              {flag.label}
            </label>
          ))}
        </div>

        <button
          type="button"
          onClick={run}
          disabled={loading}
          className="mt-5 w-full rounded-md border border-cyan-500/50 bg-cyan-500/15 px-4 py-2.5 text-sm font-semibold uppercase tracking-wider text-cyan-300 transition hover:bg-cyan-500/25 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Scoring…" : "Score exploitation risk"}
        </button>
      </section>

      <section className="glass-panel rounded-xl p-5">
        {error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        {!prediction && !error && (
          <div className="flex h-full min-h-64 flex-col items-center justify-center text-center text-sm text-slate-500">
            <p className="font-mono text-xs uppercase tracking-[0.2em] text-slate-600">
              Awaiting score
            </p>
            <p className="mt-2 max-w-sm">
              Pick a preset or tune the raw NVD/EPSS inputs, then score the
              180-day exploitation risk.
            </p>
          </div>
        )}

        {prediction && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="font-mono text-xs uppercase tracking-[0.2em] text-slate-500">
                  Calibrated 180-day KEV probability
                </p>
                <p className="mt-1 font-mono text-5xl font-semibold text-slate-50">
                  {(prediction.probability * 100).toFixed(1)}
                  <span className="ml-1 text-xl text-slate-500">%</span>
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  {prediction.cve_id} · {prediction.model_name}
                </p>
              </div>
              <span
                className={`rounded-full border px-3 py-1 text-xs font-semibold tracking-wider ${TIER_STYLES[prediction.risk_tier]}`}
              >
                {prediction.risk_tier}
              </span>
            </div>

            <div className={`rounded-lg border px-3 py-2 text-xs ${conformal.className}`}>
              {conformal.label}
            </div>

            {prediction.requires_manual_analyst_review && (
              <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                Manual analyst review required before operational action.
              </div>
            )}

            <div>
              <h3 className="text-xs uppercase tracking-wider text-slate-400">
                Local TreeSHAP waterfall
              </h3>
              {prediction.shap_base_value !== null &&
              prediction.shap_base_value !== undefined &&
              (prediction.shap_values?.length ?? 0) > 0 ? (
                <>
                  <p className="mt-1 text-[10px] text-slate-600">
                    Full base-model feature attribution; contributions are in the raw model output space.
                  </p>
                  <ShapWaterfall prediction={prediction} />
                </>
              ) : prediction.top_drivers.length === 0 ? (
                <p className="mt-2 text-xs text-slate-500">
                  TreeSHAP drivers unavailable for this model build.
                </p>
              ) : (
                <ul className="mt-3 space-y-2">
                  {prediction.top_drivers.map((driver) => (
                    <li key={driver.feature} className="flex items-center gap-3">
                      <span className="w-40 shrink-0 truncate font-mono text-[11px] text-slate-400">
                        {driver.feature}
                      </span>
                      <span className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                        <span
                          className={`block h-full rounded-full ${
                            driver.contribution >= 0
                              ? "bg-rose-500"
                              : "bg-cyan-500"
                          }`}
                          style={{
                            width: `${(Math.abs(driver.contribution) / maxDriver) * 100}%`,
                          }}
                        />
                      </span>
                      <span className="w-16 shrink-0 text-right font-mono text-[11px] text-slate-400">
                        {driver.contribution >= 0 ? "+" : ""}
                        {driver.contribution.toFixed(3)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="mt-2 text-[10px] text-slate-600">
                Red bars increase exploitation risk; blue bars decrease it.
              </p>
            </div>

            <p className="border-t border-slate-800 pt-3 text-[11px] leading-relaxed text-slate-500">
              {prediction.caveat}
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
