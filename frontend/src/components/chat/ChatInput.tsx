import { LoaderCircle, Send } from "lucide-react";
import { useState, type FormEvent, type KeyboardEvent } from "react";
import Button from "../ui/Button";

interface ChatInputProps {
  onSend: (question: string) => void;
  disabled?: boolean;
  pending?: boolean;
  placeholder?: string;
}

// Same limit as the backend's MAX_QUESTION_LENGTH default; the backend still enforces its own value.
const MAX_LENGTH = 2000;

export default function ChatInput({ onSend, disabled = false, pending = false, placeholder }: ChatInputProps) {
  const [value, setValue] = useState("");
  const canSend = !disabled && !pending && value.trim().length > 0;

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (!canSend) return;
    onSend(value.trim());
    setValue("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a new line.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form onSubmit={submit} className="flex items-end gap-2">
      <label htmlFor="chat-question" className="sr-only">
        Your question
      </label>
      <textarea
        id="chat-question"
        rows={2}
        maxLength={MAX_LENGTH}
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
        disabled={disabled}
        placeholder={placeholder ?? "Ask a question about your documents…"}
        className="max-h-40 min-h-[44px] flex-1 resize-y rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm placeholder:text-slate-400 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 focus:outline-none disabled:bg-slate-100"
      />
      <Button type="submit" disabled={!canSend} className="h-11 px-4" aria-label="Send question">
        {pending ? <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden /> : <Send className="h-4 w-4" aria-hidden />}
        <span className="hidden sm:inline">Send</span>
      </Button>
    </form>
  );
}
