import { FileText, LoaderCircle, MessageSquare, RefreshCw, Trash2 } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useChat } from "../../context/ChatContext";
import { useDocuments } from "../../context/DocumentsContext";
import { deleteDocument, getErrorMessage } from "../../services/api";
import type { DocumentStatus, DocumentSummary } from "../../types";
import { formatBytes, formatDateTime } from "../../utils/format";
import Badge, { type BadgeTone } from "../ui/Badge";
import Button from "../ui/Button";
import Card from "../ui/Card";
import { EmptyState, ErrorState, LoadingState } from "../ui/States";

export const STATUS_BADGE: Record<DocumentStatus, { label: string; tone: BadgeTone }> = {
  indexed: { label: "Indexed", tone: "success" },
  processing: { label: "Processing", tone: "info" },
  failed: { label: "Failed", tone: "danger" },
};

export default function DocumentList() {
  const { data, loading, error, refresh } = useDocuments();
  const { settings, setSettings, newChat } = useChat();
  const navigate = useNavigate();
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  async function handleDelete(doc: DocumentSummary) {
    const detail = doc.status === "indexed" ? ` Its ${doc.chunk_count} chunks will be removed from the index.` : "";
    if (!window.confirm(`Delete "${doc.filename}"?${detail}`)) return;
    setDeletingId(doc.document_id);
    setDeleteError(null);
    try {
      await deleteDocument(doc.document_id);
      // Don't leave the chat scoped to a document that no longer exists.
      if (settings.documentId === doc.document_id) setSettings({ ...settings, documentId: null });
      await refresh();
    } catch (err) {
      setDeleteError(`Could not delete ${doc.filename}: ${getErrorMessage(err)}`);
    } finally {
      setDeletingId(null);
    }
  }

  function askAbout(doc: DocumentSummary) {
    newChat();
    setSettings({ ...settings, documentId: doc.document_id });
    navigate("/chat");
  }

  const documents = data?.documents ?? [];

  return (
    <Card
      title="Documents"
      description={
        data ? `${data.indexed_count} indexed of ${data.document_count} · ${data.total_chunks} chunks` : undefined
      }
      actions={
        <Button variant="secondary" onClick={() => void refresh()} disabled={loading} aria-label="Refresh document list">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} aria-hidden /> Refresh
        </Button>
      }
    >
      {deleteError && (
        <div className="mb-4">
          <ErrorState message={deleteError} />
        </div>
      )}
      {error && <ErrorState message={error} onRetry={() => void refresh()} />}
      {!error && loading && !data && <LoadingState label="Loading documents…" />}
      {!error && data && documents.length === 0 && (
        <EmptyState icon={FileText} title="No documents yet" description="Upload a PDF to start asking questions about it." />
      )}

      {documents.length > 0 && (
        <div className="-mx-5 overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="border-b border-slate-100 text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-5 py-2 font-medium">Filename</th>
                <th className="px-5 py-2 font-medium">Document ID</th>
                <th className="px-5 py-2 text-right font-medium">Pages</th>
                <th className="px-5 py-2 text-right font-medium">Chunks</th>
                <th className="px-5 py-2 font-medium">Status</th>
                <th className="px-5 py-2 font-medium">Updated</th>
                <th className="px-5 py-2 text-right font-medium">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {documents.map((doc) => {
                const badge = STATUS_BADGE[doc.status];
                return (
                  <tr key={doc.document_id} className="align-top hover:bg-slate-50">
                    <td className="max-w-[260px] px-5 py-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 shrink-0 text-slate-400" aria-hidden />
                        <span className="truncate font-medium text-slate-900" title={doc.filename}>
                          {doc.filename}
                        </span>
                      </div>
                      {doc.file_size_bytes !== null && (
                        <p className="mt-0.5 pl-6 text-xs text-slate-400">{formatBytes(doc.file_size_bytes)}</p>
                      )}
                      {doc.error_message && <p className="mt-1 pl-6 text-xs text-red-600">{doc.error_message}</p>}
                    </td>
                    <td className="px-5 py-3 font-mono text-xs text-slate-500" title={doc.document_id}>
                      {doc.document_id.slice(0, 12)}…
                    </td>
                    <td className="px-5 py-3 text-right tabular-nums" title="Pages with text / total pages">
                      {doc.page_count === null ? doc.indexed_page_count : `${doc.indexed_page_count} / ${doc.page_count}`}
                    </td>
                    <td className="px-5 py-3 text-right tabular-nums">{doc.chunk_count}</td>
                    <td className="px-5 py-3">
                      <Badge tone={badge.tone}>{badge.label}</Badge>
                    </td>
                    <td className="px-5 py-3 text-xs whitespace-nowrap text-slate-500">{formatDateTime(doc.updated_at)}</td>
                    <td className="px-5 py-3">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          onClick={() => askAbout(doc)}
                          disabled={doc.status !== "indexed"}
                          aria-label={`Ask about ${doc.filename}`}
                        >
                          <MessageSquare className="h-4 w-4" aria-hidden /> Ask
                        </Button>
                        <Button
                          variant="danger"
                          onClick={() => void handleDelete(doc)}
                          disabled={deletingId !== null}
                          aria-label={`Delete ${doc.filename}`}
                        >
                          {deletingId === doc.document_id ? (
                            <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden />
                          ) : (
                            <Trash2 className="h-4 w-4" aria-hidden />
                          )}
                          Delete
                        </Button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
