import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { EvaluationReport, EvaluationSummary } from "../../types";
import MetricsDashboard from "./MetricsDashboard";

const summary = (ragas: EvaluationSummary["ragas"]): EvaluationSummary => ({
  cases: 35,
  answerable: 29,
  unanswerable: 6,
  errors: 0,
  answer_rate: 1,
  refusal_accuracy: 1,
  unsupported_answers: 0,
  retrieval_hit_rate: 1,
  reference_context_recall: 1,
  mrr: 0.977,
  citation_accuracy: 1,
  latency_ms: { retrieval: 20.3, rerank: null, generation: 3722.5, total: 3636.6 },
  ragas,
  ragas_complete: true,
  by_question_type: null,
});

const report: EvaluationReport = {
  report_id: "eval-20261104-124423",
  created_at: "2026-11-04T12:44:23",
  setup: { top_k: 5 },
  configs: [
    {
      name: "dense",
      retrieval_mode: "dense",
      rerank: false,
      skipped_reason: null,
      summary: summary({
        faithfulness: { mean: 1, scored: 29, missing: 0 },
        answer_relevancy: { mean: 0.836, scored: 29, missing: 0 },
      }),
    },
    {
      name: "hybrid+rerank",
      retrieval_mode: "hybrid",
      rerank: true,
      skipped_reason: "COHERE_API_KEY is not set",
      summary: null,
    },
  ],
};

describe("MetricsDashboard", () => {
  it("shows real scores and marks metrics the run did not compute", () => {
    render(<MetricsDashboard report={report} />);

    const precisionRow = screen.getByRole("row", { name: /Context precision/ });
    expect(within(precisionRow).getByText("Not computed")).toBeInTheDocument();
    const relevancyRow = screen.getByRole("row", { name: /Answer relevancy/ });
    expect(within(relevancyRow).getByText("0.84")).toBeInTheDocument();
  });

  it("explains why a configuration was skipped", () => {
    render(<MetricsDashboard report={report} />);

    expect(screen.getByText("Skipped")).toBeInTheDocument();
    expect(screen.getByText("COHERE_API_KEY is not set")).toBeInTheDocument();
  });

  it("shows an empty state when the report has no Ragas scores", () => {
    const noRagas = { ...report, configs: [{ ...report.configs[0], summary: summary({}) }] };
    render(<MetricsDashboard report={noRagas} />);

    expect(screen.getByText("No Ragas scores in this report")).toBeInTheDocument();
  });
});
