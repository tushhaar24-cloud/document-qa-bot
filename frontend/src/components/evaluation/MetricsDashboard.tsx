import { ChartColumn, TriangleAlert } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EvaluationReport } from "../../types";
import { formatMs, formatScore, humanize } from "../../utils/format";
import Badge from "../ui/Badge";
import Card from "../ui/Card";
import { EmptyState } from "../ui/States";
import {
  CONFIG_COLORS,
  RETRIEVAL_METRICS,
  completedConfigs,
  describeConfig,
  hasAnyRagasScore,
  ragasMean,
  ragasMetricNames,
} from "./metrics";

export default function MetricsDashboard({ report }: { report: EvaluationReport }) {
  const configs = completedConfigs(report);
  const metricNames = ragasMetricNames(report);

  // Recharts wants one row per x-axis group: { metric, dense: 0.84, hybrid: 0.85, ... }.
  // Missing values stay null, and Recharts simply draws no bar for them.
  const ragasData = metricNames.map((metric) => ({
    metric: humanize(metric),
    ...Object.fromEntries(configs.map((config) => [config.name, ragasMean(config, metric)])),
  }));
  const retrievalData = RETRIEVAL_METRICS.map(({ key, label }) => ({
    metric: label,
    ...Object.fromEntries(configs.map((config) => [config.name, config.summary[key] as number | null])),
  }));

  return (
    <div className="space-y-6">
      <ConfigOverview report={report} />

      <Card
        title="Ragas metrics"
        description="Scores from 0 to 1, judged by an LLM. Higher is better."
      >
        {hasAnyRagasScore(report) ? (
          <>
            <MetricChart data={ragasData} seriesNames={configs.map((c) => c.name)} />
            <RagasTable report={report} metricNames={metricNames} />
          </>
        ) : (
          <EmptyState
            icon={ChartColumn}
            title="No Ragas scores in this report"
            description="This run was made without the Ragas judge. Run the evaluation script with a judge model to get these metrics."
          />
        )}
      </Card>

      {configs.length > 0 && (
        <Card title="Retrieval & answer checks" description="Deterministic checks computed against the test dataset.">
          <MetricChart data={retrievalData} seriesNames={configs.map((c) => c.name)} />
          <LatencyTable report={report} />
        </Card>
      )}
    </div>
  );
}

function ConfigOverview({ report }: { report: EvaluationReport }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {report.configs.map((config, index) => (
        <div key={config.name} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-sm" style={{ backgroundColor: CONFIG_COLORS[index % CONFIG_COLORS.length] }} />
            <p className="font-semibold text-slate-900">{config.name}</p>
            {config.summary ? <Badge tone="success">Ran</Badge> : <Badge tone="warning">Skipped</Badge>}
          </div>
          <p className="mt-1 text-xs text-slate-500">{describeConfig(config)}</p>
          {config.summary ? (
            <p className="mt-3 text-sm text-slate-600">
              {config.summary.cases ?? "—"} questions · {config.summary.errors ?? "—"} errors
              {config.summary.ragas_complete === false && (
                <span className="mt-1 flex items-center gap-1 text-xs text-amber-700">
                  <TriangleAlert className="h-3.5 w-3.5" aria-hidden /> Some Ragas scores are missing
                </span>
              )}
            </p>
          ) : (
            <p className="mt-3 text-sm text-amber-800">{config.skipped_reason ?? "Not run."}</p>
          )}
        </div>
      ))}
    </div>
  );
}

function MetricChart({ data, seriesNames }: { data: Record<string, string | number | null>[]; seriesNames: string[] }) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis dataKey="metric" tick={{ fontSize: 12, fill: "#64748b" }} interval={0} tickLine={false} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 12, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip
            formatter={(value) => (typeof value === "number" ? value.toFixed(3) : "Not computed")}
            cursor={{ fill: "#f1f5f9" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {seriesNames.map((name, index) => (
            <Bar key={name} dataKey={name} fill={CONFIG_COLORS[index % CONFIG_COLORS.length]} radius={[4, 4, 0, 0]} maxBarSize={40} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function RagasTable({ report, metricNames }: { report: EvaluationReport; metricNames: string[] }) {
  const configs = completedConfigs(report);
  return (
    <div className="-mx-5 mt-6 overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead className="border-y border-slate-100 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-5 py-2 text-left font-medium">Metric</th>
            {configs.map((config) => (
              <th key={config.name} className="px-5 py-2 text-right font-medium">
                {config.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {metricNames.map((metric) => (
            <tr key={metric}>
              <td className="px-5 py-2 text-slate-700">{humanize(metric)}</td>
              {configs.map((config) => {
                const score = config.summary.ragas[metric];
                return (
                  <td key={config.name} className="px-5 py-2 text-right tabular-nums">
                    {score && score.mean !== null ? (
                      <>
                        <span className="font-medium text-slate-900">{formatScore(score.mean)}</span>
                        <span className="ml-1 text-xs text-slate-400">
                          ({score.scored}/{score.scored + score.missing})
                        </span>
                      </>
                    ) : (
                      <span className="text-xs text-slate-400">Not computed</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-5 pt-2 text-xs text-slate-400">(scored / answerable questions). "Not computed" means the run did not include this metric.</p>
    </div>
  );
}

function LatencyTable({ report }: { report: EvaluationReport }) {
  const configs = completedConfigs(report);
  const stages = ["retrieval", "rerank", "generation", "total"];
  if (!configs.some((config) => config.summary.latency_ms)) return null;
  return (
    <div className="-mx-5 mt-6 overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <thead className="border-y border-slate-100 text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-5 py-2 text-left font-medium">Mean latency</th>
            {configs.map((config) => (
              <th key={config.name} className="px-5 py-2 text-right font-medium">
                {config.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {stages.map((stage) => (
            <tr key={stage}>
              <td className="px-5 py-2 text-slate-700">{humanize(stage)}</td>
              {configs.map((config) => (
                <td key={config.name} className="px-5 py-2 text-right tabular-nums text-slate-900">
                  {formatMs(config.summary.latency_ms?.[stage])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
