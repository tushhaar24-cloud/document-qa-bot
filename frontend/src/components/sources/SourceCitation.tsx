import { FileText } from "lucide-react";
import type { SourceCitation as Citation } from "../../types";
import { formatScore } from "../../utils/format";

export default function SourceCitation({ source }: { source: Citation }) {
  return (
    <li id={`source-${source.chunk_id}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
        <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-indigo-600 px-1 font-semibold text-white">
          {source.citation}
        </span>
        <FileText className="h-3.5 w-3.5 text-slate-400" aria-hidden />
        <span className="max-w-[16rem] truncate font-medium text-slate-800" title={source.filename}>
          {source.filename}
        </span>
        <span className="text-slate-500">Page {source.page_number}</span>
        <span className="ml-auto flex gap-2 text-slate-400">
          <span title="Cosine similarity between the question and this passage">sim {formatScore(source.similarity)}</span>
          {source.rerank_score !== null && (
            <span title="Cohere rerank relevance (0 to 1)">rerank {formatScore(source.rerank_score)}</span>
          )}
        </span>
      </div>
      <blockquote className="mt-2 border-l-2 border-slate-300 pl-3 text-sm leading-relaxed text-slate-600">
        {source.excerpt}
      </blockquote>
    </li>
  );
}
