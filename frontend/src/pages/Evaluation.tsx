import { ChartColumn, FileUp, LoaderCircle } from "lucide-react";
import { useState } from "react";
import MetricsDashboard from "../components/evaluation/MetricsDashboard";
import Button from "../components/ui/Button";
import Card from "../components/ui/Card";
import { EmptyState, ErrorState, LoadingState } from "../components/ui/States";
import { useReport, useReportList } from "../hooks/useEvaluation";
import { getErrorMessage, importEvaluationReports } from "../services/api";
import { formatDateTime } from "../utils/format";

export default function Evaluation() {
  const reports = useReportList();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Default to the newest report until the user picks another one.
  const reportId = selectedId ?? reports.data?.[0]?.report_id ?? null;
  const report = useReport(reportId);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState<{ text: string; error: boolean } | null>(null);

  // Loads report files from backend/evaluation/reports/ that are not in the database yet.
  async function runImport() {
    setImporting(true);
    setImportMessage(null);
    try {
      const result = await importEvaluationReports();
      const parts = [`${result.imported.length} imported`, `${result.skipped.length} already stored`];
      if (result.failed.length) parts.push(`${result.failed.length} unreadable (${result.failed.join(", ")})`);
      setImportMessage({ text: `Report files: ${parts.join(", ")}.`, error: result.failed.length > 0 });
      if (result.imported.length) reports.reload();
    } catch (err) {
      setImportMessage({ text: getErrorMessage(err), error: true });
    } finally {
      setImporting(false);
    }
  }

  const importButton = (
    <Button variant="secondary" onClick={() => void runImport()} disabled={importing}>
      {importing ? <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden /> : <FileUp className="h-4 w-4" aria-hidden />}
      Import reports
    </Button>
  );

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 p-4 sm:p-6">
      {reports.error && <ErrorState message={reports.error} onRetry={reports.reload} />}
      {!reports.error && reports.loading && !reports.data && <LoadingState label="Loading evaluation reports…" />}
      {importMessage && (
        <p
          role="status"
          className={`rounded-lg px-3 py-2 text-sm ${importMessage.error ? "bg-amber-50 text-amber-800" : "bg-emerald-50 text-emerald-800"}`}
        >
          {importMessage.text}
        </p>
      )}

      {reports.data && reports.data.length === 0 && (
        <Card>
          <EmptyState
            icon={ChartColumn}
            title="No evaluation reports yet"
            description={
              <>
                Run the evaluation from the backend folder (results are saved to the database):
                <code className="mt-2 block rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">
                  python -m evaluation.evaluate_rag
                </code>
                <span className="mt-2 block">Already have report files? Import them.</span>
              </>
            }
            action={importButton}
          />
        </Card>
      )}

      {reports.data && reports.data.length > 0 && (
        <Card
          title="Evaluation report"
          description={
            report.data
              ? `${String(report.data.setup.dataset ?? "Dataset")} · ${formatDateTime(report.data.created_at)}`
              : undefined
          }
          actions={
            <>
              {importButton}
              <label htmlFor="report-select" className="sr-only">
                Evaluation report
              </label>
              <select
                id="report-select"
                value={reportId ?? ""}
                onChange={(event) => setSelectedId(event.target.value)}
                className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              >
                {reports.data.map((info) => (
                  <option key={info.report_id} value={info.report_id}>
                    {formatDateTime(info.created_at)} — {info.config_names.join(", ")}
                  </option>
                ))}
              </select>
            </>
          }
        >
          {report.data ? <SetupList setup={report.data.setup} /> : <p className="text-sm text-slate-500">—</p>}
        </Card>
      )}

      {report.error && <ErrorState message={report.error} onRetry={report.reload} />}
      {!report.error && report.loading && <LoadingState label="Loading report…" />}
      {!report.error && !report.loading && report.data && <MetricsDashboard report={report.data} />}
    </div>
  );
}

function SetupList({ setup }: { setup: Record<string, unknown> }) {
  const entries = Object.entries(setup);
  if (entries.length === 0) return <p className="text-sm text-slate-500">No setup details recorded.</p>;
  return (
    <details>
      <summary className="cursor-pointer text-sm font-medium text-indigo-700">Run setup ({entries.length} settings)</summary>
      <dl className="mt-3 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
        {entries.map(([key, value]) => (
          <div key={key} className="min-w-0">
            <dt className="text-xs text-slate-500">{key}</dt>
            <dd className="break-words text-slate-800">{String(value)}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
