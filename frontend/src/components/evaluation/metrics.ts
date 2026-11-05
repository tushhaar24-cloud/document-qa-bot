// Helpers that turn an evaluation report into chart/table rows.
// A metric missing from the report stays null: it is shown as "Not computed",
// never replaced by 0 or an estimate.

import type { EvaluationConfigResult, EvaluationReport, EvaluationSummary } from "../../types";

// The four core Ragas metrics, always listed (even when a run did not compute them).
export const CORE_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] as const;

export const RETRIEVAL_METRICS: { key: keyof EvaluationSummary; label: string }[] = [
  { key: "answer_rate", label: "Answer rate" },
  { key: "refusal_accuracy", label: "Refusal accuracy" },
  { key: "retrieval_hit_rate", label: "Retrieval hit rate" },
  { key: "reference_context_recall", label: "Reference passages" },
  { key: "mrr", label: "MRR" },
  { key: "citation_accuracy", label: "Citation accuracy" },
];

export const CONFIG_COLORS = ["#4f46e5", "#059669", "#d97706", "#64748b"];

/** Configurations that actually ran (skipped ones have no summary). */
export function completedConfigs(report: EvaluationReport): (EvaluationConfigResult & { summary: EvaluationSummary })[] {
  return report.configs.filter((config): config is EvaluationConfigResult & { summary: EvaluationSummary } =>
    Boolean(config.summary),
  );
}

/** Core metrics first, then any extra metric found in the report (e.g. factual_correctness). */
export function ragasMetricNames(report: EvaluationReport): string[] {
  const extra = new Set<string>();
  for (const config of completedConfigs(report)) {
    for (const name of Object.keys(config.summary.ragas)) {
      if (!(CORE_RAGAS_METRICS as readonly string[]).includes(name)) extra.add(name);
    }
  }
  return [...CORE_RAGAS_METRICS, ...[...extra].sort()];
}

export function ragasMean(config: EvaluationConfigResult, metric: string): number | null {
  return config.summary?.ragas[metric]?.mean ?? null;
}

export function hasAnyRagasScore(report: EvaluationReport): boolean {
  return completedConfigs(report).some((config) =>
    Object.values(config.summary.ragas).some((score) => score.mean !== null),
  );
}

export function describeConfig(config: EvaluationConfigResult): string {
  const parts = [config.retrieval_mode ? `${config.retrieval_mode} retrieval` : null];
  if (config.rerank !== null) parts.push(config.rerank ? "Cohere rerank" : "no rerank");
  return parts.filter(Boolean).join(" · ") || "single run";
}
