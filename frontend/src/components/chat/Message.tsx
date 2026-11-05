import { Bot, CircleAlert, SearchX, User } from "lucide-react";
import type { ReactNode } from "react";
import type { ChatMessage } from "../../types";
import { formatMs, formatTime } from "../../utils/format";
import SourceCitation from "../sources/SourceCitation";
import Badge from "../ui/Badge";

/** Show the answer's [n] citation markers as small badges that match the source list. */
function renderAnswer(text: string): ReactNode[] {
  return text.split(/(\[\d+\])/g).map((part, index) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) return part;
    return (
      <sup
        key={index}
        className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-indigo-100 px-1 text-[10px] font-semibold text-indigo-700"
      >
        {match[1]}
      </sup>
    );
  });
}

export default function Message({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end gap-3">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600 px-4 py-2.5 text-sm whitespace-pre-wrap text-white sm:max-w-[70%]">
          {message.content}
        </div>
        <Avatar>
          <User className="h-4 w-4" aria-hidden />
        </Avatar>
      </div>
    );
  }

  const { response, error } = message;
  return (
    <div className="flex gap-3">
      <Avatar assistant>
        <Bot className="h-4 w-4" aria-hidden />
      </Avatar>
      <div className="min-w-0 max-w-[85%] flex-1 space-y-3 sm:max-w-[80%]">
        {error && (
          <div role="alert" className="flex items-start gap-2 rounded-2xl rounded-tl-sm border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
            <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
            <span>{error}</span>
          </div>
        )}

        {response?.status === "not_found" && (
          <div className="rounded-2xl rounded-tl-sm border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
            <p className="flex items-center gap-2 font-medium">
              <SearchX className="h-4 w-4" aria-hidden /> No supporting evidence found
            </p>
            <p className="mt-1 whitespace-pre-wrap">{response.answer}</p>
          </div>
        )}

        {response?.status === "answered" && (
          <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-slate-800 shadow-sm">
            {renderAnswer(response.answer)}
          </div>
        )}

        {response && response.sources.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium tracking-wide text-slate-500 uppercase">
              Sources ({response.sources.length})
            </p>
            <ul className="space-y-2">
              {response.sources.map((source) => (
                <SourceCitation key={`${source.citation}-${source.chunk_id}`} source={source} />
              ))}
            </ul>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span>{formatTime(message.createdAt)}</span>
          {response && (
            <>
              <Badge>{response.retrieval_mode}</Badge>
              <Badge tone={response.reranking.status === "applied" ? "info" : response.reranking.status === "failed" ? "warning" : "neutral"}>
                rerank: {response.reranking.status}
                {response.reranking.latency_ms !== null && ` (${formatMs(response.reranking.latency_ms)})`}
              </Badge>
              <span>
                {response.context_chunk_count} of {response.retrieved_chunk_count} retrieved passages used
              </span>
              {!response.history_saved && <Badge tone="warning">Not saved to history</Badge>}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Avatar({ children, assistant = false }: { children: ReactNode; assistant?: boolean }) {
  return (
    <span
      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
        assistant ? "bg-indigo-50 text-indigo-600" : "bg-slate-200 text-slate-600"
      }`}
    >
      {children}
    </span>
  );
}
