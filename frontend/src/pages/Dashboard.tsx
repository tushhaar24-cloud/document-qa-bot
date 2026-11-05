import { ChartColumn, CircleAlert, CircleCheck, Database, FileText, Layers, LoaderCircle, MessageSquare } from "lucide-react";
import { Link } from "react-router-dom";
import { CORE_RAGAS_METRICS, completedConfigs, ragasMean } from "../components/evaluation/metrics";
import Card from "../components/ui/Card";
import StatCard from "../components/ui/StatCard";
import { EmptyState, ErrorState, LoadingState } from "../components/ui/States";
import { useChat } from "../context/ChatContext";
import { useDocuments } from "../context/DocumentsContext";
import { useReport, useReportList } from "../hooks/useEvaluation";
import type { DocumentSummary } from "../types";
import { formatDateTime, formatScore, humanize } from "../utils/format";

export default function Dashboard() {
  const { data, loading, error, refresh } = useDocuments();
  const { conversationTotal, conversationsLoading } = useChat();
  const indexed = data?.documents.filter((doc) => doc.status === "indexed");
  const indexedPages = indexed?.reduce((sum, doc) => sum + doc.indexed_page_count, 0);
  const notIndexed = data ? data.document_count - data.indexed_count : 0;
  const show = (value: number | undefined | null, busy = loading) => (value === undefined || value === null ? (busy ? "…" : "—") : value);

  return (
    <div className="mx-auto w-full max-w-7xl space-y-6 p-4 sm:p-6">
      {error && <ErrorState message={error} onRetry={() => void refresh()} />}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Indexed documents"
          value={show(data?.indexed_count)}
          hint={notIndexed > 0 ? `${notIndexed} more failed or still processing` : "Searchable in the vector store"}
          icon={FileText}
        />
        <StatCard label="Indexed pages" value={show(indexedPages)} hint="Pages that produced text chunks" icon={Layers} />
        <StatCard label="Total chunks" value={show(data?.total_chunks)} hint="Passages stored in ChromaDB" icon={Database} />
        <StatCard
          label="Saved conversations"
          value={show(conversationTotal, conversationsLoading)}
          hint="Chat history in PostgreSQL"
          icon={MessageSquare}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Recent document activity" description="Latest uploads and their processing status.">
          {data && data.documents.length === 0 ? (
            <EmptyState
              icon={FileText}
              title="No documents yet"
              description={
                <>
                  <Link to="/documents" className="font-medium text-indigo-700 hover:underline">
                    Upload a PDF
                  </Link>{" "}
                  to see it here.
                </>
              }
            />
          ) : !data ? (
            loading ? <LoadingState /> : <p className="text-sm text-slate-500">—</p>
          ) : (
            <ul className="divide-y divide-slate-100">
              {data.documents.slice(0, 8).map((doc) => (
                <ActivityRow key={doc.document_id} doc={doc} />
              ))}
            </ul>
          )}
        </Card>

        <EvaluationSummaryCard />
      </div>
    </div>
  );
}

function ActivityRow({ doc }: { doc: DocumentSummary }) {
  const [icon, verb, detail] =
    doc.status === "indexed"
      ? [<CircleCheck className="h-4 w-4 text-emerald-500" aria-hidden />, "Indexed", `${doc.chunk_count} chunks from ${doc.indexed_page_count} pages`]
      : doc.status === "failed"
        ? [<CircleAlert className="h-4 w-4 text-red-500" aria-hidden />, "Failed", doc.error_message ?? "Processing failed"]
        : [<LoaderCircle className="h-4 w-4 text-indigo-500" aria-hidden />, "Processing", "Extracting text and building embeddings"];
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className="mt-0.5">{icon}</span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-slate-900">
          <span className="font-medium">{verb}</span> {doc.filename}
        </p>
        <p className="truncate text-xs text-slate-500">{detail}</p>
      </div>
      <span className="shrink-0 text-xs text-slate-400">{formatDateTime(doc.updated_at)}</span>
    </li>
  );
}

function EvaluationSummaryCard() {
  const reports = useReportList();
  const latestId = reports.data?.[0]?.report_id ?? null;
  const report = useReport(latestId);
  const configs = report.data ? completedConfigs(report.data) : [];

  return (
    <Card
      title="Latest evaluation"
      description={report.data ? formatDateTime(report.data.created_at) : "Ragas scores from the newest saved report."}
      actions={
        <Link to="/evaluation" className="text-sm font-medium text-indigo-700 hover:underline">
          View details
        </Link>
      }
    >
      {(reports.error || report.error) && (
        <ErrorState message={(reports.error ?? report.error) as string} onRetry={reports.error ? reports.reload : report.reload} />
      )}
      {!reports.error && !report.error && (reports.loading || report.loading) && <LoadingState />}
      {reports.data && reports.data.length === 0 && (
        <EmptyState icon={ChartColumn} title="No evaluation reports yet" description="Run the backend evaluation script to see metrics here." />
      )}
      {report.data && !report.loading && configs.length > 0 && (
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs uppercase tracking-wide text-slate-500">
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
              {CORE_RAGAS_METRICS.map((metric) => (
                <tr key={metric}>
                  <td className="px-5 py-2 text-slate-700">{humanize(metric)}</td>
                  {configs.map((config) => {
                    const mean = ragasMean(config, metric);
                    return (
                      <td key={config.name} className="px-5 py-2 text-right tabular-nums">
                        {mean === null ? <span className="text-xs text-slate-400">n/a</span> : formatScore(mean)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
