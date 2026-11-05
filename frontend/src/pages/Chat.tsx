import { History, LoaderCircle, Plus, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import ChatInput from "../components/chat/ChatInput";
import ChatWindow from "../components/chat/ChatWindow";
import ConversationList from "../components/chat/ConversationList";
import Button from "../components/ui/Button";
import { useChat } from "../context/ChatContext";
import { useDocuments } from "../context/DocumentsContext";
import type { RetrievalMode } from "../types";

const SELECT_CLASS = "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm sm:w-auto";

export default function Chat() {
  const { messages, pending, settings, setSettings, send, newChat, openingConversation, conversationId } = useChat();
  const { data, error } = useDocuments();
  const [historyOpen, setHistoryOpen] = useState(false);
  // Only indexed documents can be searched.
  const documents = (data?.documents ?? []).filter((doc) => doc.status === "indexed");
  const noDocuments = data !== null && documents.length === 0;
  const scopedDoc = documents.find((doc) => doc.document_id === settings.documentId);
  // A saved conversation may be scoped to a document that has since been deleted.
  const scopeMissing = settings.documentId !== null && data !== null && !scopedDoc;

  return (
    <div className="flex min-h-0 flex-1">
      {/* Conversation history: a column on large screens, a drawer on small ones */}
      <aside className="hidden w-72 shrink-0 border-r border-slate-200 bg-white lg:block">
        <ConversationList />
      </aside>
      {historyOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden" role="dialog" aria-label="Conversation history">
          <div className="w-80 max-w-[85%] bg-white shadow-xl">
            <div className="flex items-center justify-end p-2">
              <button
                type="button"
                className="rounded p-1 text-slate-500 hover:bg-slate-100"
                aria-label="Close history"
                onClick={() => setHistoryOpen(false)}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <ConversationList onSelect={() => setHistoryOpen(false)} />
          </div>
          <div className="flex-1 bg-slate-900/40" onClick={() => setHistoryOpen(false)} aria-hidden />
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Settings bar */}
        <div className="flex flex-col gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:flex-row sm:flex-wrap sm:items-end sm:px-6">
          <Field label="Search in" htmlFor="chat-document">
            <select
              id="chat-document"
              value={settings.documentId ?? ""}
              onChange={(event) => setSettings({ ...settings, documentId: event.target.value || null })}
              className={`${SELECT_CLASS} sm:max-w-xs`}
            >
              <option value="">All documents ({documents.length})</option>
              {documents.map((doc) => (
                <option key={doc.document_id} value={doc.document_id}>
                  {doc.filename}
                </option>
              ))}
              {scopeMissing && (
                <option value={settings.documentId ?? ""} disabled>
                  Deleted document
                </option>
              )}
            </select>
          </Field>
          <Field label="Retrieval" htmlFor="chat-retrieval">
            <select
              id="chat-retrieval"
              value={settings.retrievalMode ?? ""}
              onChange={(event) =>
                setSettings({ ...settings, retrievalMode: (event.target.value || null) as RetrievalMode | null })
              }
              className={SELECT_CLASS}
            >
              <option value="">Server default</option>
              <option value="hybrid">Hybrid (dense + BM25)</option>
              <option value="dense">Dense (embeddings)</option>
              <option value="sparse">Sparse (BM25)</option>
            </select>
          </Field>
          <Field label="Reranking" htmlFor="chat-rerank">
            <select
              id="chat-rerank"
              value={settings.rerank === null ? "" : String(settings.rerank)}
              onChange={(event) =>
                setSettings({ ...settings, rerank: event.target.value === "" ? null : event.target.value === "true" })
              }
              className={SELECT_CLASS}
            >
              <option value="">Server default</option>
              <option value="true">On (Cohere)</option>
              <option value="false">Off</option>
            </select>
          </Field>
          <div className="flex gap-2 sm:ml-auto">
            <Button variant="secondary" onClick={() => setHistoryOpen(true)} className="lg:hidden">
              <History className="h-4 w-4" aria-hidden /> History
            </Button>
            {/* On large screens the history column has its own New chat button. */}
            <Button
              variant="ghost"
              onClick={newChat}
              disabled={(messages.length === 0 && !conversationId) || pending}
              className="lg:hidden"
            >
              <Plus className="h-4 w-4" aria-hidden /> New chat
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6">
          <div className="mx-auto max-w-4xl">
            {error && (
              <p className="mb-4 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
                Could not load the document list ({error}). You can still ask across all documents.
              </p>
            )}
            {scopeMissing && (
              <p className="mb-4 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
                This conversation was limited to a document that has been deleted. Choose another document to keep asking.
              </p>
            )}
            {openingConversation ? (
              <div role="status" className="flex items-center justify-center gap-2 py-10 text-sm text-slate-500">
                <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden /> Loading conversation…
              </div>
            ) : noDocuments && messages.length === 0 ? (
              <div className="py-10 text-center text-sm text-slate-500">
                No documents are indexed yet.{" "}
                <Link to="/documents" className="font-medium text-indigo-700 hover:underline">
                  Upload a PDF
                </Link>{" "}
                to start asking questions.
              </div>
            ) : (
              <ChatWindow
                messages={messages}
                pending={pending}
                emptyHint={
                  scopedDoc
                    ? `Questions will be answered from “${scopedDoc.filename}” only.`
                    : "Questions are answered from all indexed documents, with page-level citations."
                }
              />
            )}
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-slate-200 bg-white px-4 py-3 sm:px-6">
          <div className="mx-auto max-w-4xl">
            <ChatInput
              onSend={(question) => void send(question)}
              pending={pending}
              disabled={noDocuments || openingConversation || scopeMissing}
            />
            <p className="mt-1.5 text-xs text-slate-400">
              Enter to send · Shift+Enter for a new line · Each question is answered from the documents on its own
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, htmlFor, children }: { label: string; htmlFor: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={htmlFor} className="text-xs font-medium text-slate-500">
        {label}
      </label>
      {children}
    </div>
  );
}
