"use client";

import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

import { fetchRecentTriage } from "@/lib/api";
import type { TriageItem } from "@/lib/types";

type QueueFilter = "all" | "priority" | "manual";

function percentage(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "—";
}

function badgeStyle(conformal: TriageItem["conformal_set"]): string {
  if (conformal === "{0,1}") return "border-amber-500/40 bg-amber-500/10 text-amber-200";
  if (conformal === "{1}") return "border-rose-500/40 bg-rose-500/10 text-rose-200";
  return "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
}

export default function LiveTriageQueue({
  onInspect,
}: {
  onInspect: (item: TriageItem) => void;
}) {
  const [items, setItems] = useState<TriageItem[]>([]);
  const [total, setTotal] = useState(0);
  const [cutoff, setCutoff] = useState<string | null>(null);
  const [filter, setFilter] = useState<QueueFilter>("all");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchRecentTriage(25)
      .then((response) => {
        if (!active) return;
        setItems(response.items);
        setTotal(response.total_unmatured);
        setCutoff(response.cutoff_date ?? null);
      })
      .catch((caught: unknown) => {
        if (active) setError(caught instanceof Error ? caught.message : "Unable to load triage queue");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const visibleItems = useMemo(() => {
    const query = search.trim().toLowerCase();
    return items.filter((item) => {
      if (filter === "priority" && !["CRITICAL", "HIGH"].includes(item.risk_tier)) return false;
      if (filter === "manual" && !item.requires_manual_analyst_review && item.conformal_set !== "{0,1}") return false;
      return !query || [item.cve_id, item.top_shap_driver ?? ""].some((value) => value.toLowerCase().includes(query));
    });
  }, [filter, items, search]);

  return (
    <section className="glass-panel rounded-xl p-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-cyan-400">
            Unmatured CVEs
          </p>
          <h2 className="mt-1 text-lg font-semibold text-slate-100">Live triage queue</h2>
          <p className="mt-1 text-xs text-slate-500">
            {total.toLocaleString("en-US")} right-censored records · censoring cutoff{" "}
            {cutoff ? cutoff.slice(0, 10) : "unavailable"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="relative">
            <span className="sr-only">Search CVE or top SHAP driver</span>
            <Search size={14} className="pointer-events-none absolute left-2.5 top-2.5 text-slate-500" />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search CVE / driver"
              className="w-48 rounded-lg border border-slate-700 bg-slate-950/70 py-2 pl-8 pr-3 text-xs text-slate-200 outline-none transition focus:border-cyan-500/60"
            />
          </label>
          <select
            aria-label="Filter triage queue"
            value={filter}
            onChange={(event) => setFilter(event.target.value as QueueFilter)}
            className="rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-slate-300 outline-none focus:border-cyan-500/60"
          >
            <option value="all">All</option>
            <option value="priority">High Priority</option>
            <option value="manual">Manual Review {`{0,1}`}</option>
          </select>
        </div>
      </div>

      {loading && <p className="py-12 text-center text-sm text-slate-500">Loading ranked unmatured CVEs…</p>}
      {error && <p role="alert" className="mt-5 rounded-lg border border-rose-500/30 bg-rose-500/5 p-3 text-sm text-rose-200">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <div className="mt-5 rounded-lg border border-slate-800 bg-slate-950/50 p-8 text-center">
          <p className="font-mono text-xs uppercase tracking-wider text-slate-400">Queue is empty</p>
          <p className="mt-2 text-xs text-slate-600">No unmatured_recent.parquet artifact is present or it contains no right-censored CVEs.</p>
        </div>
      )}
      {!loading && !error && items.length > 0 && (
        <div className="mt-5 overflow-x-auto">
          <table className="min-w-[1050px] w-full text-left text-xs">
            <thead className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-3 py-3">CVE</th>
                <th className="px-3 py-3">Published</th>
                <th className="px-3 py-3">CVSS</th>
                <th className="px-3 py-3">Anchored EPSS</th>
                <th className="px-3 py-3">180d KEV risk</th>
                <th className="px-3 py-3">Conformal</th>
                <th className="px-3 py-3">Top SHAP driver</th>
                <th className="px-3 py-3">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {visibleItems.map((item) => (
                <tr key={item.cve_id} className="transition hover:bg-cyan-500/[0.03]">
                  <td className="px-3 py-3 font-mono text-cyan-200">{item.cve_id}</td>
                  <td className="px-3 py-3 font-mono text-slate-400">{item.published_date?.slice(0, 10) ?? "—"}</td>
                  <td className="px-3 py-3 font-mono text-slate-300">{item.cvss_base_score?.toFixed(1) ?? "—"}</td>
                  <td className="px-3 py-3 font-mono text-slate-300">{percentage(item.epss)}</td>
                  <td className="px-3 py-3 font-mono font-semibold text-slate-100">{percentage(item.probability)}</td>
                  <td className="px-3 py-3">
                    <span className={`rounded-md border px-2 py-1 font-mono text-[10px] ${badgeStyle(item.conformal_set)}`}>
                      {item.conformal_set}
                    </span>
                    {item.requires_manual_analyst_review && (
                      <span className="ml-1 rounded-md border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-[9px] text-orange-200">
                        REVIEW
                      </span>
                    )}
                  </td>
                  <td className="max-w-48 truncate px-3 py-3 font-mono text-slate-400">{item.top_shap_driver ?? "Unavailable"}</td>
                  <td className="px-3 py-3">
                    <button
                      type="button"
                      onClick={() => onInspect(item)}
                      className="whitespace-nowrap rounded-md border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1.5 text-[10px] font-medium text-cyan-200 transition hover:bg-cyan-500/20"
                    >
                      Inspect in Risk Scorer
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {visibleItems.length === 0 && (
            <p className="py-10 text-center text-xs text-slate-500">No CVEs match the current search and filter.</p>
          )}
        </div>
      )}
    </section>
  );
}
