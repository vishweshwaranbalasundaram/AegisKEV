"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fetchBenchmark } from "@/lib/api";
import type { BenchmarkResult, ModelMetrics } from "@/lib/types";

const CHART_COLORS = ["#22d3ee", "#a78bfa", "#fb7185"];
const TOOLTIP_STYLE = {
  backgroundColor: "#0b1220",
  border: "1px solid rgba(34, 211, 238, 0.25)",
  borderRadius: 8,
  color: "#e2e8f0",
  fontSize: 12,
};

function decimal(value?: number): string {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

function percent(value?: number): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="glass-panel rounded-xl p-5">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
        {title}
      </h2>
      <div className="mt-4 h-80 w-full">{children}</div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-700/70 bg-slate-950/60 p-3">
      <p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-2xl text-slate-100">{value}</p>
      {hint && <p className="mt-0.5 text-[10px] text-slate-600">{hint}</p>}
    </div>
  );
}

function ModelTable({
  models,
  champion,
}: {
  models: Record<string, ModelMetrics>;
  champion?: string;
}) {
  const names = Object.keys(models).sort(
    (left, right) => (models[right]?.pr_auc ?? 0) - (models[left]?.pr_auc ?? 0),
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-left text-xs">
        <thead className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
          <tr>
            <th className="py-2 pr-4 font-medium">Model</th>
            <th className="py-2 pr-4 font-medium">PR-AUC</th>
            <th className="py-2 pr-4 font-medium">ROC-AUC</th>
            <th className="py-2 pr-4 font-medium">Recall@1%</th>
            <th className="py-2 pr-4 font-medium">RVC@100</th>
            <th className="py-2 pr-4 font-medium">RVC@500</th>
            <th className="py-2 pr-4 font-medium">ECE</th>
            <th className="py-2 font-medium">Brier</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-900">
          {names.map((name) => {
            const metrics = models[name] ?? {};
            const isChampion = name === champion;
            return (
              <tr key={name} className={isChampion ? "bg-cyan-500/5" : undefined}>
                <td className="py-2 pr-4 font-mono text-[11px] text-slate-300">
                  {isChampion && <span className="mr-1 text-cyan-400">★</span>}
                  {name}
                </td>
                <td className="py-2 pr-4 font-mono text-slate-200">{decimal(metrics.pr_auc)}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{decimal(metrics.roc_auc)}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{percent(metrics.recall_top_1pct)}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{decimal(metrics.rvc_100)}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{decimal(metrics.rvc_500)}</td>
                <td className="py-2 pr-4 font-mono text-slate-400">{decimal(metrics.ece_10)}</td>
                <td className="py-2 font-mono text-slate-400">{decimal(metrics.brier)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarkPanel() {
  const [result, setResult] = useState<BenchmarkResult | null>(null);

  useEffect(() => {
    let active = true;
    fetchBenchmark().then((next) => {
      if (active) setResult(next);
    });
    return () => {
      active = false;
    };
  }, []);

  if (result === null) {
    return (
      <div className="glass-panel rounded-xl p-6 text-sm text-slate-400">
        Loading verified benchmark artifacts…
      </div>
    );
  }

  if (!result.ok) {
    return (
      <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-amber-300">
          Benchmark not populated
        </h2>
        <p className="mt-2 max-w-2xl text-sm text-slate-400">{result.detail}</p>
        <pre className="mt-4 overflow-x-auto rounded-lg border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-cyan-300">
          python -m ml.training.train_pipeline --trials 25 --n-boot 1000
        </pre>
        <p className="mt-3 text-xs text-slate-500">
          This view renders only the JSON emitted by the training pipeline.
        </p>
      </div>
    );
  }

  const data = result.data;
  const models = data.models ?? {};
  const ablation = data.ablation ?? {};
  const champion = data.champion?.name;
  const baseline = models.Rule_EPSS_Only;
  const modelChartData = Object.entries(models).map(([name, metrics]) => ({
    name,
    "PR-AUC": (metrics.pr_auc ?? 0) * 100,
    "Recall @ Top-1%": (metrics.recall_top_1pct ?? 0) * 100,
    "RVC @ 500": (metrics.rvc_500 ?? 0) * 100,
  }));
  const stageChartData = Object.entries(ablation)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, metrics]) => ({
      name: name.replace(/^Stage_(\d)_/, "Stage $1 · ").replaceAll("_", " "),
      "PR-AUC": (metrics.pr_auc ?? 0) * 100,
      "Recall @ Top-1%": (metrics.recall_top_1pct ?? 0) * 100,
      "RVC @ 500": (metrics.rvc_500 ?? 0) * 100,
    }));
  const shapData = (data.shap_top_features ?? [])
    .slice(0, 15)
    .map(({ feature, mean_abs_shap }) => ({ feature, value: mean_abs_shap }));
  const modelNames = ["PR-AUC", "Recall @ Top-1%", "RVC @ 500"];

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="Champion"
          value={champion ?? "—"}
          hint={data.champion?.selection_metric ?? "validation selection"}
        />
        <MetricCard
          label="Champion PR-AUC"
          value={decimal(data.champion?.val_hpo_pr_auc)}
          hint="val_hpo, untouched by test"
        />
        <MetricCard
          label="Conformal coverage"
          value={percent(data.conformal?.coverage)}
          hint={`alpha ${data.conformal?.alpha ?? "—"} · q̂ ${decimal(data.conformal?.q_hat)}`}
        />
        <MetricCard
          label="Ambiguous sets"
          value={percent(data.conformal?.ambiguous_fraction)}
          hint="share of test rows with {0,1}"
        />
      </div>

      <ChartCard title="Model comparison · held-out test">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={modelChartData} margin={{ top: 8, right: 12, left: 0, bottom: 60 }}>
            <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" angle={-30} textAnchor="end" interval={0} height={72} tick={{ fill: "#94a3b8", fontSize: 10 }} />
            <YAxis tickFormatter={(value: number) => `${value}%`} tick={{ fill: "#64748b", fontSize: 10 }} />
            <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => `${Number(value ?? 0).toFixed(2)}%`} />
            {modelNames.map((name, index) => (
              <Bar key={name} dataKey={name} fill={CHART_COLORS[index]} radius={[3, 3, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Three-stage feature ablation">
        {stageChartData.length === 0 ? (
          <p className="pt-8 text-sm text-slate-500">No ablation results in metrics_summary.json.</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={stageChartData} margin={{ top: 8, right: 20, left: 0, bottom: 42 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" angle={-12} textAnchor="end" height={56} tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis tickFormatter={(value: number) => `${value}%`} tick={{ fill: "#64748b", fontSize: 10 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => `${Number(value ?? 0).toFixed(2)}%`} />
              <Bar dataKey="PR-AUC" fill={CHART_COLORS[0]} radius={[3, 3, 0, 0]} />
              <Line dataKey="Recall @ Top-1%" stroke={CHART_COLORS[1]} strokeWidth={2} dot={{ r: 4 }} />
              <Line dataKey="RVC @ 500" stroke={CHART_COLORS[2]} strokeWidth={2} dot={{ r: 4 }} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      <ChartCard title="Global top TreeSHAP drivers">
        {shapData.length === 0 ? (
          <p className="pt-8 text-sm text-slate-500">No global SHAP results in metrics_summary.json.</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shapData} layout="vertical" margin={{ top: 4, right: 24, left: 80, bottom: 4 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis dataKey="feature" type="category" width={150} tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => Number(value ?? 0).toFixed(4)} />
              <Bar dataKey="value" name="Mean |SHAP|" fill="#22d3ee" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </ChartCard>

      <section className="glass-panel rounded-xl p-5">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Held-out test benchmark
          </h2>
          <p className="text-[10px] text-slate-600">
            {data.experiment ?? "AegisKEV"} · generated {data.generated_at ?? "unknown"}
          </p>
        </div>
        <div className="mt-4"><ModelTable models={models} champion={champion} /></div>
        <p className="mt-3 text-[10px] text-slate-600">
          RVC = risk-weighted vulnerability capture using CVSS base score. Recall@1% and RVC use the untouched test split.
          {baseline ? ` EPSS baseline PR-AUC: ${decimal(baseline.pr_auc)}.` : ""}
        </p>
      </section>
    </div>
  );
}
