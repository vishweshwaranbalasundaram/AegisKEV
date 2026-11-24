const SAFEGUARDS = [
  {
    title: "EPSS anchored at publication + 1 day",
    body: "EPSS features come from the archived daily snapshot at NVD published + 1 day (+2 fallback), never from the KEV dateAdded row. Using dateAdded would leak the post-KEV EPSS spike and erase true negatives.",
  },
  {
    title: "Target defined only forward in time",
    body: "y_180d = 1 iff anchor < KEV dateAdded ≤ anchor + 180d. Rows already in KEV at triage (dateAdded ≤ anchor) are excluded instead of being labelled.",
  },
  {
    title: "Right-censoring instead of guessing",
    body: "kev_cutoff = max(dateAdded). Anchors after kev_cutoff − 180d are written to unmatured_recent.parquet and never enter a labelled split, so a recent row cannot be mislabelled as a negative.",
  },
  {
    title: "Causal priors reveal only after maturation",
    body: "A CVE's CWE/vendor KEV prior becomes visible only when current_day ≥ anchor + 180d. Same-day rows are scored as one batch, and priors are computed on matured rows before splitting.",
  },
  {
    title: "Chronological, non-overlapping splits",
    body: "train ≤ 2024-06-30, val_hpo 2024-07-01…2024-11-30, val_cal 2024-12-01…2025-03-16, test ≥ 2025-03-17. A quantile fallback keeps four contiguous blocks on reduced year subsets.",
  },
  {
    title: "Fit-on-train-only text pipelines",
    body: "TF-IDF/SVD and embedding PCA are fit on train only, then applied to validation and test. No vocabulary or projection learns from future data.",
  },
  {
    title: "Sealed test split + no cisa columns",
    body: "Model selection uses val_hpo PR-AUC; test is scored once. All cisa* columns and date_added are dropped before feature frames, enforced by finalize_features and assert_no_cisa_columns.",
  },
];

export default function MethodologyCard() {
  return (
    <div className="space-y-6">
      <section className="glass-panel rounded-xl p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
          Anti-leakage architecture
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          This is a point-in-time forecaster, not a retrospective classifier.
          Every feature at triage time could have been observed on that day, and
          the label only exists in the following 180 days. The seven safeguards
          below are enforced in code and covered by the leakage test suite.
        </p>

        <ol className="mt-6 space-y-4">
          {SAFEGUARDS.map((safeguard, index) => (
            <li
              key={safeguard.title}
              className="flex gap-4 rounded-lg border border-slate-800 bg-slate-950/60 p-4"
            >
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-cyan-500/40 bg-cyan-500/10 font-mono text-xs text-cyan-300">
                {index + 1}
              </span>
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {safeguard.title}
                </p>
                <p className="mt-1 text-xs leading-relaxed text-slate-400">
                  {safeguard.body}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="glass-panel rounded-xl p-5">
          <h3 className="text-xs uppercase tracking-wider text-slate-400">
            Evaluation contract
          </h3>
          <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-400">
            <li>· PR-AUC, ROC-AUC, Brier, 10-bin ECE on the untouched test split.</li>
            <li>· Recall / Precision / F1 and TP·FP·FN·TN at the top 1% ranked rows.</li>
            <li>· RVC@100 and RVC@500 weight captured positives by CVSS base score.</li>
            <li>· 1,000-resample paired bootstrap over the same rows for ΔPR-AUC vs Rule_EPSS_Only and LogisticRegression.</li>
          </ul>
        </div>
        <div className="glass-panel rounded-xl p-5">
          <h3 className="text-xs uppercase tracking-wider text-slate-400">
            Uncertainty &amp; review policy
          </h3>
          <ul className="mt-3 space-y-2 text-xs leading-relaxed text-slate-400">
            <li>· Probabilities are Platt-calibrated on val_cal with a frozen estimator.</li>
            <li>· Split conformal at α = 0.10 yields {"{0}"}, {"{1}"} or {"{0,1}"} at α = 90%.</li>
            <li>· {"{0,1}"} (ambiguous) predictions always require manual analyst review.</li>
            <li>· CRITICAL and HIGH tiers are flagged for review regardless of set size.</li>
          </ul>
        </div>
      </section>

      <section className="glass-panel rounded-xl p-5">
        <h3 className="text-xs uppercase tracking-wider text-slate-400">
          What this tool is not
        </h3>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          It is not a severity score, not a patch-priority oracle, and not a
          guarantee of exploitation. A low probability means the historical
          evidence resembles CVEs that did not reach KEV within 180 days — it
          does not mean a vulnerability is safe. Operational decisions must
          combine this forecast with asset exposure and vendor advisories.
        </p>
      </section>
    </div>
  );
}
