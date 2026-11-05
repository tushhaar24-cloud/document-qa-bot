import type { ConversationDetail, DocumentListResponse, DocumentUploadResponse, QuestionResponse } from "../types";

export const DOC_ID = "0123456789abcdef0123456789abcdef";

export const CONVERSATION_ID = "6f1c2a52-8d3e-4c55-9b8a-2f7d1e0c4a11";

export const documentList: DocumentListResponse = {
  document_count: 2,
  indexed_count: 1,
  total_chunks: 12,
  documents: [
    {
      document_id: DOC_ID,
      filename: "handbook.pdf",
      status: "indexed",
      error_message: null,
      chunk_count: 12,
      indexed_page_count: 6,
      page_count: 6,
      file_size_bytes: 52_000,
      created_at: "2026-11-04T10:00:00+00:00",
      updated_at: "2026-11-04T10:00:05+00:00",
      indexed_at: "2026-11-04T10:00:05+00:00",
    },
    {
      document_id: "f".repeat(32),
      filename: "broken.pdf",
      status: "failed",
      error_message: "The file is not a valid PDF.",
      chunk_count: 0,
      indexed_page_count: 0,
      page_count: null,
      file_size_bytes: 24,
      created_at: "2026-11-04T09:00:00+00:00",
      updated_at: "2026-11-04T09:00:00+00:00",
      indexed_at: null,
    },
  ],
};

export const emptyDocumentList: DocumentListResponse = { document_count: 0, indexed_count: 0, total_chunks: 0, documents: [] };

export const uploadResult: DocumentUploadResponse = {
  document_id: DOC_ID,
  filename: "handbook.pdf",
  page_count: 6,
  pages_with_text: 6,
  empty_pages: [],
  total_characters: 9000,
  pages: [],
  chunk_count: 12,
  indexing_status: "indexed",
  indexed_chunk_count: 12,
  chunk_size: 800,
  chunk_overlap: 120,
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  embedding_dimensions: 384,
};

export const answered: QuestionResponse = {
  question: "What is the notice period?",
  answer: "Employees must give two months of written notice [1].",
  status: "answered",
  document_id: null,
  sources: [
    {
      citation: 1,
      chunk_id: "chunk-1",
      document_id: DOC_ID,
      filename: "handbook.pdf",
      page_number: 6,
      excerpt: "After the first year, the notice period is two months.",
      similarity: 0.71,
      rerank_score: null,
    },
  ],
  retrieved_chunk_count: 5,
  context_chunk_count: 3,
  retrieval_mode: "hybrid",
  reranking: { status: "disabled", model: null, latency_ms: null, error: null },
  conversation_id: CONVERSATION_ID,
  message_id: "a1b2c3d4-0000-4000-8000-000000000002",
  history_saved: true,
};

export const notFound: QuestionResponse = {
  ...answered,
  answer: "I could not find this in the uploaded documents.",
  status: "not_found",
  sources: [],
  context_chunk_count: 0,
};

export const savedConversation: ConversationDetail = {
  id: CONVERSATION_ID,
  title: "What is the notice period?",
  document_id: null,
  created_at: "2026-11-04T10:05:00+00:00",
  updated_at: "2026-11-04T10:05:03+00:00",
  messages: [
    {
      id: "a1b2c3d4-0000-4000-8000-000000000001",
      role: "user",
      content: "What is the notice period?",
      created_at: "2026-11-04T10:05:00+00:00",
      status: null,
      document_id: null,
      retrieval_mode: null,
      rerank_status: null,
      rerank_model: null,
      retrieved_chunk_count: null,
      context_chunk_count: null,
      sources: [],
    },
    {
      id: "a1b2c3d4-0000-4000-8000-000000000002",
      role: "assistant",
      content: answered.answer,
      created_at: "2026-11-04T10:05:03+00:00",
      status: "answered",
      document_id: null,
      retrieval_mode: "hybrid",
      rerank_status: "disabled",
      rerank_model: null,
      retrieved_chunk_count: 5,
      context_chunk_count: 3,
      sources: answered.sources,
    },
  ],
};
