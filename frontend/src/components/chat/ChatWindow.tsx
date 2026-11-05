import { Bot, LoaderCircle, MessageSquare } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../../types";
import { EmptyState } from "../ui/States";
import Message from "./Message";

interface ChatWindowProps {
  messages: ChatMessage[];
  pending: boolean;
  emptyHint: string;
}

export default function ChatWindow({ messages, pending, emptyHint }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  if (messages.length === 0 && !pending) {
    return <EmptyState icon={MessageSquare} title="Ask your first question" description={emptyHint} />;
  }

  return (
    <div className="space-y-6" aria-live="polite">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      {pending && (
        <div className="flex gap-3" role="status">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-50 text-indigo-600">
            <Bot className="h-4 w-4" aria-hidden />
          </span>
          <div className="flex items-center gap-2 rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
            <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden />
            Searching documents and generating an answer…
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
