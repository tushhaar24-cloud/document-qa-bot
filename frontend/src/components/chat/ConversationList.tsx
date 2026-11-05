import { MessageSquare, Pencil, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { useChat } from "../../context/ChatContext";
import { getErrorMessage } from "../../services/api";
import { formatDateTime } from "../../utils/format";
import Button from "../ui/Button";
import { ErrorState, LoadingState } from "../ui/States";

/** Saved conversations: open, rename, delete, or start a new chat. */
export default function ConversationList({ onSelect }: { onSelect?: () => void }) {
  const {
    conversations, conversationsLoading, conversationsError, refreshConversations,
    conversationId, openConversation, newChat, removeConversation, renameConversation, pending,
  } = useChat();
  const [actionError, setActionError] = useState<string | null>(null);

  async function run(action: () => Promise<void>) {
    setActionError(null);
    try {
      await action();
    } catch (err) {
      setActionError(getErrorMessage(err));
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="p-3">
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => {
            newChat();
            onSelect?.();
          }}
        >
          <Plus className="h-4 w-4" aria-hidden /> New chat
        </Button>
      </div>
      <p className="px-4 pb-1 text-xs font-medium tracking-wide text-slate-500 uppercase">History</p>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {actionError && (
          <div className="mb-2 px-1">
            <ErrorState message={actionError} />
          </div>
        )}
        {conversationsError && <ErrorState message={conversationsError} onRetry={() => void refreshConversations()} />}
        {!conversationsError && conversationsLoading && conversations.length === 0 && <LoadingState label="Loading…" />}
        {!conversationsError && !conversationsLoading && conversations.length === 0 && (
          <p className="px-2 py-4 text-sm text-slate-500">Saved conversations will appear here.</p>
        )}
        <ul className="space-y-0.5" aria-label="Saved conversations">
          {conversations.map((conversation) => {
            const active = conversation.id === conversationId;
            return (
              <li key={conversation.id} className="group relative">
                <button
                  type="button"
                  disabled={pending}
                  onClick={() => {
                    void openConversation(conversation.id);
                    onSelect?.();
                  }}
                  aria-current={active ? "true" : undefined}
                  className={`flex w-full items-start gap-2 rounded-lg py-2 pr-16 pl-2 text-left text-sm disabled:cursor-not-allowed ${
                    active ? "bg-indigo-50 text-indigo-800" : "text-slate-700 hover:bg-slate-100"
                  }`}
                >
                  <MessageSquare className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" aria-hidden />
                  <span className="min-w-0">
                    <span className="block truncate font-medium">{conversation.title}</span>
                    <span className="block text-xs text-slate-400">
                      {formatDateTime(conversation.updated_at)} · {conversation.message_count} messages
                    </span>
                  </span>
                </button>
                <div className="absolute top-1.5 right-1 flex opacity-100 sm:opacity-0 sm:group-focus-within:opacity-100 sm:group-hover:opacity-100">
                  <button
                    type="button"
                    className="rounded p-1.5 text-slate-400 hover:bg-white hover:text-slate-700"
                    aria-label={`Rename ${conversation.title}`}
                    onClick={() => {
                      const title = window.prompt("Rename conversation", conversation.title)?.trim();
                      if (title && title !== conversation.title) void run(() => renameConversation(conversation.id, title));
                    }}
                  >
                    <Pencil className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    className="rounded p-1.5 text-slate-400 hover:bg-white hover:text-red-600"
                    aria-label={`Delete ${conversation.title}`}
                    onClick={() => {
                      if (window.confirm(`Delete "${conversation.title}" and all its messages?`)) {
                        void run(() => removeConversation(conversation.id));
                      }
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
