// Chat state shared across pages: the open conversation, its messages, the list of
// saved conversations and the search settings. Conversations are stored by the
// backend (PostgreSQL), so they survive page reloads.

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import * as api from "../services/api";
import { getErrorMessage } from "../services/api";
import type { ChatMessage, ConversationDetail, ConversationSummary, QuestionResponse, RetrievalMode } from "../types";

export interface ChatSettings {
  documentId: string | null; // null = search every document
  retrievalMode: RetrievalMode | null; // null = backend default
  rerank: boolean | null; // null = backend default
}

interface ChatState {
  messages: ChatMessage[];
  pending: boolean;
  settings: ChatSettings;
  setSettings: (settings: ChatSettings) => void;
  send: (question: string) => Promise<void>;
  // Saved conversations
  conversationId: string | null;
  conversations: ConversationSummary[]; // the most recent 50
  conversationTotal: number | null; // all saved conversations; null until loaded
  conversationsLoading: boolean;
  conversationsError: string | null;
  refreshConversations: () => Promise<void>;
  openConversation: (id: string) => Promise<void>;
  openingConversation: boolean;
  newChat: () => void;
  removeConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, title: string) => Promise<void>;
}

const ChatContext = createContext<ChatState | null>(null);

const DEFAULT_SETTINGS: ChatSettings = { documentId: null, retrievalMode: null, rerank: null };

/** Rebuild chat messages from a stored conversation, in the same shape live answers use. */
export function toChatMessages(conversation: ConversationDetail): ChatMessage[] {
  let lastQuestion = "";
  return conversation.messages.map((message) => {
    const base = { id: message.id, role: message.role, content: message.content, createdAt: Date.parse(message.created_at) };
    if (message.role === "user") {
      lastQuestion = message.content;
      return base;
    }
    const response: QuestionResponse = {
      question: lastQuestion,
      answer: message.content,
      status: message.status ?? "answered",
      document_id: message.document_id,
      sources: message.sources,
      retrieved_chunk_count: message.retrieved_chunk_count ?? 0,
      context_chunk_count: message.context_chunk_count ?? 0,
      retrieval_mode: message.retrieval_mode ?? "hybrid",
      // Latency is not stored, so it is not shown for saved answers.
      reranking: { status: message.rerank_status ?? "disabled", model: message.rerank_model, latency_ms: null, error: null },
      conversation_id: conversation.id,
      message_id: message.id,
      history_saved: true,
    };
    return { ...base, response };
  });
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>(DEFAULT_SETTINGS);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationTotal, setConversationTotal] = useState<number | null>(null);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [conversationsError, setConversationsError] = useState<string | null>(null);
  const [openingConversation, setOpeningConversation] = useState(false);
  // Bumped whenever the open conversation changes, so a late answer for the old one is dropped.
  const generation = useRef(0);

  const refreshConversations = useCallback(async () => {
    setConversationsLoading(true);
    try {
      const result = await api.listConversations();
      setConversations(result.conversations);
      setConversationTotal(result.total);
      setConversationsError(null);
    } catch (err) {
      setConversationsError(getErrorMessage(err));
    } finally {
      setConversationsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || pending) return;
      const turn = generation.current;
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: text, createdAt: Date.now() }]);
      setPending(true);
      try {
        const response = await api.askQuestion({
          question: text,
          conversation_id: conversationId,
          document_id: settings.documentId,
          retrieval_mode: settings.retrievalMode,
          rerank: settings.rerank,
        });
        if (turn !== generation.current) return;
        setMessages((current) => [
          ...current,
          { id: response.message_id ?? crypto.randomUUID(), role: "assistant", content: response.answer, createdAt: Date.now(), response },
        ]);
        if (response.history_saved && response.conversation_id) {
          setConversationId(response.conversation_id);
          void refreshConversations();
        }
      } catch (err) {
        if (turn !== generation.current) return;
        setMessages((current) => [
          ...current,
          { id: crypto.randomUUID(), role: "assistant", content: "", createdAt: Date.now(), error: getErrorMessage(err) },
        ]);
      } finally {
        if (turn === generation.current) setPending(false);
      }
    },
    [pending, settings, conversationId, refreshConversations],
  );

  const newChat = useCallback(() => {
    generation.current += 1;
    setConversationId(null);
    setMessages([]);
    setPending(false);
  }, []);

  const openConversation = useCallback(async (id: string) => {
    generation.current += 1;
    const turn = generation.current;
    setPending(false);
    setOpeningConversation(true);
    try {
      const conversation = await api.getConversation(id);
      if (turn !== generation.current) return;
      setConversationId(conversation.id);
      setMessages(toChatMessages(conversation));
      setSettings((current) => ({ ...current, documentId: conversation.document_id }));
    } catch (err) {
      if (turn !== generation.current) return;
      setConversationId(null);
      setMessages([{ id: crypto.randomUUID(), role: "assistant", content: "", createdAt: Date.now(), error: getErrorMessage(err) }]);
    } finally {
      if (turn === generation.current) setOpeningConversation(false);
    }
  }, []);

  const removeConversation = useCallback(
    async (id: string) => {
      await api.deleteConversation(id);
      if (id === conversationId) newChat();
      await refreshConversations();
    },
    [conversationId, newChat, refreshConversations],
  );

  const renameConversation = useCallback(
    async (id: string, title: string) => {
      await api.renameConversation(id, title);
      await refreshConversations();
    },
    [refreshConversations],
  );

  const value = useMemo(
    () => ({
      messages, pending, settings, setSettings, send,
      conversationId, conversations, conversationTotal, conversationsLoading, conversationsError, refreshConversations,
      openConversation, openingConversation, newChat, removeConversation, renameConversation,
    }),
    [
      messages, pending, settings, send,
      conversationId, conversations, conversationTotal, conversationsLoading, conversationsError, refreshConversations,
      openConversation, openingConversation, newChat, removeConversation, renameConversation,
    ],
  );
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatState {
  const context = useContext(ChatContext);
  if (!context) throw new Error("useChat must be used inside <ChatProvider>");
  return context;
}
