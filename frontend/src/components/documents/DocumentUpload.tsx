// PDF upload with a per-file queue. The backend accepts one file per request,
// so several selected files are uploaded one after another, each with its own
// progress bar and status.

import { CircleAlert, CircleCheck, FileUp, LoaderCircle, X } from "lucide-react";
import { useRef, useState, type DragEvent } from "react";
import { useDocuments } from "../../context/DocumentsContext";
import { getErrorMessage, uploadDocument } from "../../services/api";
import type { DocumentUploadResponse } from "../../types";
import { formatBytes } from "../../utils/format";
import Badge from "../ui/Badge";
import Button from "../ui/Button";
import Card from "../ui/Card";

type UploadStatus = "queued" | "uploading" | "processing" | "indexed" | "failed";

interface UploadItem {
  id: string;
  file: File;
  status: UploadStatus;
  progress: number;
  result?: DocumentUploadResponse;
  error?: string;
}

const STATUS_LABEL: Record<UploadStatus, string> = {
  queued: "Queued",
  uploading: "Uploading",
  processing: "Extracting & indexing",
  indexed: "Indexed",
  failed: "Failed",
};

function isPdf(file: File): boolean {
  return file.name.toLowerCase().endsWith(".pdf");
}

export default function DocumentUpload() {
  const { refresh } = useDocuments();
  const [items, setItems] = useState<UploadItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const update = (id: string, changes: Partial<UploadItem>) =>
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...changes } : item)));

  async function uploadAll(files: File[]) {
    if (files.length === 0) return;
    const queued: UploadItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      // Reject obvious non-PDFs before sending; the backend checks again.
      status: isPdf(file) ? "queued" : "failed",
      progress: 0,
      error: isPdf(file) ? undefined : "Only .pdf files are accepted.",
    }));
    setItems((current) => [...queued, ...current]);
    setBusy(true);

    let sent = 0;
    for (const item of queued) {
      if (item.status === "failed") continue;
      update(item.id, { status: "uploading" });
      sent += 1;
      try {
        const result = await uploadDocument(item.file, (percent) =>
          // Once every byte is sent, the server is still extracting text and building embeddings.
          update(item.id, { progress: percent, status: percent >= 100 ? "processing" : "uploading" }),
        );
        update(item.id, { status: "indexed", progress: 100, result });
      } catch (err) {
        update(item.id, { status: "failed", error: getErrorMessage(err) });
      }
    }

    setBusy(false);
    // Failed uploads are recorded by the backend too, so refresh after any attempt.
    if (sent > 0) await refresh();
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    if (!busy) void uploadAll(Array.from(event.dataTransfer.files));
  }

  const finished = items.some((item) => item.status === "indexed" || item.status === "failed");

  return (
    <Card
      title="Upload PDFs"
      description="Text is extracted, chunked, embedded and indexed so it can be searched."
      actions={
        finished && !busy ? (
          <Button
            variant="ghost"
            onClick={() => setItems((current) => current.filter((item) => item.status !== "indexed" && item.status !== "failed"))}
          >
            Clear finished
          </Button>
        ) : null
      }
    >
      <label
        htmlFor="pdf-upload"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-8 text-center ${
          dragging ? "border-indigo-400 bg-indigo-50" : "border-slate-300 hover:border-indigo-300 hover:bg-slate-50"
        } ${busy ? "pointer-events-none opacity-60" : ""}`}
      >
        <FileUp className="h-8 w-8 text-indigo-500" aria-hidden />
        <span className="mt-2 text-sm font-medium text-slate-900">Drop PDF files here or click to browse</span>
        <span className="mt-1 text-xs text-slate-500">Multiple files are uploaded one at a time.</span>
        <input
          ref={inputRef}
          id="pdf-upload"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          disabled={busy}
          className="sr-only"
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            // Reset so choosing the same file again still triggers onChange.
            event.target.value = "";
            void uploadAll(files);
          }}
        />
      </label>

      {items.length > 0 && (
        <ul className="mt-4 divide-y divide-slate-100" aria-label="Upload queue">
          {items.map((item) => (
            <li key={item.id} className="py-3">
              <div className="flex items-center gap-3">
                <StatusIcon status={item.status} />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="truncate text-sm font-medium text-slate-900">{item.file.name}</p>
                    <span className="text-xs text-slate-400">{formatBytes(item.file.size)}</span>
                  </div>
                  {item.result && (
                    <p className="mt-0.5 text-xs text-slate-500">
                      {item.result.page_count} pages · {item.result.indexed_chunk_count} chunks ·{" "}
                      <span className="font-mono">{item.result.document_id}</span>
                      {item.result.empty_pages.length > 0 &&
                        ` · ${item.result.empty_pages.length} page(s) had no extractable text`}
                    </p>
                  )}
                  {item.error && <p className="mt-0.5 text-xs text-red-600">{item.error}</p>}
                </div>
                <Badge tone={item.status === "indexed" ? "success" : item.status === "failed" ? "danger" : "info"}>
                  {STATUS_LABEL[item.status]}
                </Badge>
                {!busy && (item.status === "indexed" || item.status === "failed") && (
                  <button
                    type="button"
                    className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                    aria-label={`Dismiss ${item.file.name}`}
                    onClick={() => setItems((current) => current.filter((other) => other.id !== item.id))}
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
              {(item.status === "uploading" || item.status === "processing") && (
                <div
                  className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100"
                  role="progressbar"
                  aria-label={`Upload progress for ${item.file.name}`}
                  aria-valuenow={item.progress}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className={`h-full rounded-full bg-indigo-500 ${item.status === "processing" ? "animate-pulse" : ""}`}
                    style={{ width: `${item.progress}%` }}
                  />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

function StatusIcon({ status }: { status: UploadStatus }) {
  if (status === "indexed") return <CircleCheck className="h-5 w-5 shrink-0 text-emerald-500" aria-hidden />;
  if (status === "failed") return <CircleAlert className="h-5 w-5 shrink-0 text-red-500" aria-hidden />;
  if (status === "queued") return <FileUp className="h-5 w-5 shrink-0 text-slate-400" aria-hidden />;
  return <LoaderCircle className="h-5 w-5 shrink-0 animate-spin text-indigo-500" aria-hidden />;
}
