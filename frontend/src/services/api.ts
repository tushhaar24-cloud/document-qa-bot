// Centralized HTTP client for the FastAPI backend.
// UI components never call axios directly: they call the functions below, so
// endpoint paths, timeouts and error handling live in exactly one place.

import axios, { type AxiosProgressEvent } from "axios";
import type {
  ConversationDetail,
  ConversationList,
  DocumentDeleteResponse,
  DocumentListResponse,
  DocumentUploadResponse,
  EvaluationImportResponse,
  EvaluationReport,
  EvaluationReportList,
  QuestionRequest,
  QuestionResponse,
} from "../types";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8030").replace(/\/+$/, "");

export const api = axios.create({
  baseURL: API_BASE_URL,
  // Answers wait on the LLM (backend gives Groq up to 30 s), so allow more than that.
  timeout: 60_000,
});

// Embedding a large PDF runs on the backend CPU and can take minutes.
const UPLOAD_TIMEOUT_MS = 5 * 60_000;

// --- Documents ---

export async function listDocuments(): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>("/api/documents");
  return data;
}

export async function uploadDocument(
  file: File,
  onProgress?: (percent: number) => void,
): Promise<DocumentUploadResponse> {
  // The browser sets the multipart boundary itself when the body is FormData.
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<DocumentUploadResponse>("/api/documents/upload", form, {
    timeout: UPLOAD_TIMEOUT_MS,
    onUploadProgress: (event: AxiosProgressEvent) => {
      if (onProgress && event.total) onProgress(Math.round((event.loaded / event.total) * 100));
    },
  });
  return data;
}

export async function deleteDocument(documentId: string): Promise<DocumentDeleteResponse> {
  const { data } = await api.delete<DocumentDeleteResponse>(`/api/documents/${encodeURIComponent(documentId)}`);
  return data;
}

// --- Question answering ---

export async function askQuestion(request: QuestionRequest): Promise<QuestionResponse> {
  // Leave optional fields out instead of sending null, so the backend applies its own defaults.
  const body = Object.fromEntries(Object.entries(request).filter(([, value]) => value !== undefined && value !== null));
  const { data } = await api.post<QuestionResponse>("/api/ask", body);
  return data;
}

// --- Chat history ---

export async function listConversations(limit = 50, offset = 0): Promise<ConversationList> {
  const { data } = await api.get<ConversationList>("/api/conversations", { params: { limit, offset } });
  return data;
}

export async function getConversation(conversationId: string): Promise<ConversationDetail> {
  const { data } = await api.get<ConversationDetail>(`/api/conversations/${encodeURIComponent(conversationId)}`);
  return data;
}

export async function renameConversation(conversationId: string, title: string): Promise<ConversationDetail> {
  const { data } = await api.patch<ConversationDetail>(`/api/conversations/${encodeURIComponent(conversationId)}`, {
    title,
  });
  return data;
}

export async function deleteConversation(conversationId: string): Promise<void> {
  await api.delete(`/api/conversations/${encodeURIComponent(conversationId)}`);
}

// --- Evaluation reports ---

export async function listEvaluationReports(): Promise<EvaluationReportList> {
  const { data } = await api.get<EvaluationReportList>("/api/evaluation/reports");
  return data;
}

export async function getEvaluationReport(reportId: string): Promise<EvaluationReport> {
  const { data } = await api.get<EvaluationReport>(`/api/evaluation/reports/${encodeURIComponent(reportId)}`);
  return data;
}

export async function importEvaluationReports(): Promise<EvaluationImportResponse> {
  const { data } = await api.post<EvaluationImportResponse>("/api/evaluation/reports/import");
  return data;
}

// --- Errors ---

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Turn any thrown value into a message that is safe and useful to show a user. */
export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.code === "ECONNABORTED" || error.code === "ETIMEDOUT") {
      return "The request timed out. The server may still be busy; please try again.";
    }
    if (!error.response) {
      return `Cannot reach the backend at ${API_BASE_URL}. Check that the API server is running.`;
    }
    // FastAPI puts its message in `detail`: a string for HTTPException, a list for validation errors.
    const detail = (error.response.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string" && detail) return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return (detail as ValidationIssue[])
        .map((issue) => {
          const field = issue.loc?.filter((part) => part !== "body").join(".");
          return field ? `${field}: ${issue.msg}` : issue.msg;
        })
        .join("; ");
    }
    return `Request failed with status ${error.response.status}.`;
  }
  if (error instanceof Error && error.message) return error.message;
  return "Something went wrong.";
}
