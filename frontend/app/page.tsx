"use client";

import { useEffect, useState } from "react";
import { Activity, Download, FileText } from "lucide-react";

import BenchmarkPanel from "@/components/BenchmarkPanel";
import LiveTriageQueue from "@/components/LiveTriageQueue";
import MethodologyCard from "@/components/MethodologyCard";
import TriagePanel from "@/components/TriagePanel";
import { API_BASE, fetchBenchmark } from "@/lib/api";
import type { Benchmark, BenchmarkResult, TriageItem } from "@/lib/types";

type View = "triage" | "benchmark" | "methodology" | "queue";

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "triage", label: "Risk Scorer", hint: "Score a CVE" },
  { id: "benchmark", label: "Benchmark", hint: "Verified metrics" },
  { id: "methodology", label: "Methodology", hint: "Anti-leakage" },
  { id: "queue", label: "Live Triage Queue", hint: "Unmatured CVEs" },
];

function KpiCard({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <article className="glass-panel min-h-32 rounded-xl p-4">
      <p className="text-[10px] uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-2 font-mono text-2xl font-semibold text-slate-100">{value}</p>
      <p className="mt-1 text-[11px] text-slate-500">{note}</p>
    </article>
  );
}

function ExecutiveKpis({ result }: { result: BenchmarkResult | null }) {
  if (!result?.ok) {
    return (
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {["Best Holdout PR-AUC", "Top-1% Triage Capture", "Calibration & Conformal", "Temporal Corpus Scale"].map((label) => (
          <KpiCard key={label} label={label} value="—" note="Verified benchmark artifact unavailable" />
        ))}
      </div>
    );
  }
  const data: Benchmark = result.data;
  const modelEntries = Object.entries(data.models ?? {});
  const best = [...modelEntries].sort(
    ([, left], [, right]) => (right.pr_auc ?? 0) - (left.pr_auc ?? 0),
  )[0];
  const [bestName, bestMetrics] = best ?? ["", {}];
  const baseline = data.models?.Rule_EPSS_Only;
  const multiplier =
    bestMetrics.pr_auc && baseline?.pr_auc
      ? `${(bestMetrics.pr_auc / baseline.pr_auc).toFixed(1)}x`
      : "—";
  const pValue = bestMetrics.bootstrap_vs_rule_epss?.p_value_one_sided;
  const recallMultiplier =
    bestMetrics.recall_top_1pct && baseline?.recall_top_1pct
      ? `${(bestMetrics.recall_top_1pct / baseline.recall_top_1pct).toFixed(1)}x capture`
      : "—";
  const championName = data.champion?.name;
  const championEce = championName ? data.models?.[championName]?.ece_10 : undefined;
  const rows = data.dataset_manifest?.row_counts ?? data.dataset?.rows ?? {};
  const matured = Object.values(rows).reduce((sum, value) => sum + value, 0);
  const unmatured = data.dataset_manifest?.unmatured_rows ?? 0;
  const curated = matured + unmatured;
  const testRows = rows.test;

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <KpiCard
        label="Best Holdout PR-AUC"
        value={bestMetrics.pr_auc?.toFixed(3) ?? "—"}
        note={`${bestName || "—"} · ${multiplier} vs EPSS${pValue !== undefined ? ` · p = ${pValue.toFixed(3)}` : ""}`}
      />
      <KpiCard
        label="Top-1% Triage Capture"
        value={bestMetrics.recall_top_1pct !== undefined ? `${(bestMetrics.recall_top_1pct * 100).toFixed(1)}%` : "—"}
        note={`${baseline?.recall_top_1pct !== undefined ? `${(baseline.recall_top_1pct * 100).toFixed(1)}% EPSS baseline` : "EPSS baseline unavailable"} · ${recallMultiplier}`}
      />
      <KpiCard
        label="Calibration & Conformal"
        value={championEce !== undefined ? `ECE ${championEce.toFixed(3)}` : "ECE —"}
        note={`Champion ${championName ?? "—"} · ${data.conformal?.coverage !== undefined ? `${(data.conformal.coverage * 100).toFixed(1)}% coverage` : "coverage unavailable"} · alpha ${data.conformal?.alpha ?? "—"}`}
      />
      <KpiCard
        label="Temporal Corpus Scale"
        value={curated ? curated.toLocaleString("en-US") : "—"}
        note={`${testRows !== undefined ? testRows.toLocaleString("en-US") : "—"} held-out test rows`}
      />
    </div>
  );
}

export default function Page() {
  const [view, setView] = useState<View>("triage");
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [queuedCve, setQueuedCve] = useState<TriageItem | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchBenchmark().then((result) => {
      if (active) setBenchmark(result);
    });
    return () => {
      active = false;
    };
  }, []);

  async function download(path: string, filename: string) {
    setDownloadError(null);
    try {
      const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail ?? `Download failed: HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (caught) {
      setDownloadError(caught instanceof Error ? caught.message : "Download failed");
    }
  }

  function inspectInScorer(item: TriageItem) {
    setQueuedCve(item);
    setView("triage");
  }

  return (
    <main className="mx-auto max-w-[1500px] px-4 py-6 sm:px-6 lg:px-8">
      <header className="flex flex-wrap items-end justify-between gap-5 border-b border-slate-800/80 pb-6">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-cyan-400">
            AegisKEV // SOC Analyst Workbench
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
            Exploitation Intelligence
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            Point-in-time 180-day CISA KEV forecasting with calibrated probability,
            conformal uncertainty, and local TreeSHAP explanations.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="mr-2 flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-300">
            <Activity size={14} />
            Inference API · 180d
          </div>
          <button
            type="button"
            onClick={() => void download("/api/v1/reports/markdown", "AegisKEV_Evaluation_Report.md")}
            className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs font-medium text-cyan-200 transition hover:border-cyan-300/60 hover:bg-cyan-500/20"
          >
            <FileText size={14} /> Export Report (.md)
          </button>
          <button
            type="button"
            onClick={() => void download("/api/v1/metrics/benchmark", "AegisKEV_Metrics.json")}
            className="inline-flex items-center gap-2 rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-xs font-medium text-violet-200 transition hover:border-violet-300/60 hover:bg-violet-500/20"
          >
            <Download size={14} /> Download Metrics (.json)
          </button>
        </div>
      </header>

      {downloadError && (
        <p role="alert" className="mt-3 text-xs text-rose-300">{downloadError}</p>
      )}
      <section aria-label="Executive KPIs" className="mt-5">
        <ExecutiveKpis result={benchmark} />
      </section>

      <nav aria-label="Workbench views" className="mt-6 flex flex-wrap gap-2">
        {VIEWS.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-current={view === item.id ? "page" : undefined}
            onClick={() => setView(item.id)}
            className={`rounded-lg border px-4 py-2 text-left transition-all duration-200 ${
              view === item.id
                ? "border-cyan-400/50 bg-cyan-400/10 shadow-[0_0_20px_rgba(34,211,238,0.08)]"
                : "border-slate-800 bg-slate-900/40 hover:border-slate-600"
            }`}
          >
            <span className={`block text-sm font-medium ${view === item.id ? "text-cyan-200" : "text-slate-300"}`}>
              {item.label}
            </span>
            <span className="block font-mono text-[10px] uppercase tracking-wider text-slate-500">
              {item.hint}
            </span>
          </button>
        ))}
      </nav>

      <div className="mt-5">
        {view === "triage" && <TriagePanel queuedCve={queuedCve} />}
        {view === "benchmark" && <BenchmarkPanel />}
        {view === "methodology" && <MethodologyCard />}
        {view === "queue" && <LiveTriageQueue onInspect={inspectInScorer} />}
      </div>

      <footer className="mt-10 border-t border-slate-800 pt-4 text-[10px] leading-relaxed text-slate-600">
        Forecast only. Not a severity score or guarantee of exploitation. Ambiguous
        conformal sets and HIGH/CRITICAL predictions require manual analyst review.
        Metrics and queue items are loaded from generated local artifacts.
      </footer>
    </main>
  );
}
