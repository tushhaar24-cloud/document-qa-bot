// TypeScript mirrors of the FastAPI Pydantic schemas (backend/app/schemas/*.py).
// Keep field names identical to the backend so responses need no conversion.

// --- Shared (schemas/common.py) ---

export type RetrievalMode = "dense" | "sparse" | "hybrid";

export interface RerankInfo {
  status: "applied" | "disabled" | "skipped" | "failed";
  model: string | null;
  latency_ms: number | null;
  error: string | null;
}

// --- Documents (schemas/document.py) ---

export interface PageSummary {
  page_number: number;
  char_count: number;
  text_preview: string | null;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  page_count: number;
  pages_with_text: number;
  empty_pages: number[];
  total_characters: number;
  pages: PageSummary[];
  chunk_count: number;
  indexing_status: "indexed";
  indexed_chunk_count: number;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  embedding_dimensions: number;
}

export type DocumentStatus = "processing" | "indexed" | "failed";

export interface DocumentSummary {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  error_message: string | null;
  chunk_count: number;
  indexed_page_count: number;
  page_count: number | null;
  file_size_bytes: number | null;
  created_at: string;
  updated_at: string;
  indexed_at: string | null;
}

export interface DocumentListResponse {
  document_count: number; // every document, whatever its status
  indexed_count: number;
  total_chunks: number;
  documents: DocumentSummary[]; // most recently changed first
}

export interface DocumentDeleteResponse {
  document_id: string;
  deleted_chunks: number;
}

// --- Questions (schemas/question.py) ---

export interface QuestionRequest {
  question: string;
  // Continue a saved conversation; omit to start a new one.
  conversation_id?: string | null;
  // Omit to search every indexed document.
  document_id?: string | null;
  top_k?: number;
  // Omit to use the backend's RETRIEVAL_MODE / RERANK_ENABLED settings.
  retrieval_mode?: RetrievalMode | null;
  rerank?: boolean | null;
}

export interface SourceCitation {
  citation: number; // matches the [n] marker in the answer
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  excerpt: string;
  similarity: number;
  rerank_score: number | null;
}

export interface QuestionResponse {
  question: string;
  answer: string;
  status: "answered" | "not_found";
  document_id: string | null;
  sources: SourceCitation[];
  retrieved_chunk_count: number;
  context_chunk_count: number;
  retrieval_mode: RetrievalMode;
  reranking: RerankInfo;
  // Where the exchange was saved; history_saved is false when saving failed.
  conversation_id: string | null;
  message_id: string | null;
  history_saved: boolean;
}

// --- Chat history (schemas/conversation.py) ---

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  status: "answered" | "not_found" | null;
  document_id: string | null;
  retrieval_mode: RetrievalMode | null;
  rerank_status: RerankInfo["status"] | null;
  rerank_model: string | null;
  retrieved_chunk_count: number | null;
  context_chunk_count: number | null;
  sources: SourceCitation[];
}

export interface ConversationSummary {
  id: string;
  title: string;
  document_id: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail {
  id: string;
  title: string;
  document_id: string | null;
  created_at: string;
  updated_at: string;
  messages: StoredMessage[];
}

export interface ConversationList {
  total: number;
  conversations: ConversationSummary[];
}

// --- Evaluation reports (schemas/evaluation.py) ---

export interface RagasScore {
  mean: number | null;
  scored: number;
  missing: number;
}

export interface EvaluationSummary {
  cases: number | null;
  answerable: number | null;
  unanswerable: number | null;
  errors: number | null;
  answer_rate: number | null;
  refusal_accuracy: number | null;
  unsupported_answers: number | null;
  retrieval_hit_rate: number | null;
  reference_context_recall: number | null;
  mrr: number | null;
  citation_accuracy: number | null;
  latency_ms: Record<string, number | null> | null;
  ragas: Record<string, RagasScore>;
  ragas_complete: boolean | null;
  by_question_type: Record<string, { cases: number; passed: number }> | null;
}

export interface EvaluationConfigResult {
  name: string;
  retrieval_mode: string | null;
  rerank: boolean | null;
  skipped_reason: string | null;
  summary: EvaluationSummary | null;
}

export interface EvaluationReport {
  report_id: string;
  created_at: string;
  setup: Record<string, unknown>;
  configs: EvaluationConfigResult[];
}

export interface EvaluationReportInfo {
  report_id: string;
  created_at: string;
  config_names: string[];
  ragas_metrics: string[];
}

export interface EvaluationReportList {
  reports: EvaluationReportInfo[];
}

export interface EvaluationImportResponse {
  imported: string[];
  skipped: string[];
  failed: string[];
}

// --- Frontend-only state ---

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  // Assistant messages only:
  response?: QuestionResponse;
  error?: string;
}
