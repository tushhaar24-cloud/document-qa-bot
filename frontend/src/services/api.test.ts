import { AxiosError, AxiosHeaders, type AxiosAdapter, type InternalAxiosRequestConfig } from "axios";
import { afterEach, describe, expect, it } from "vitest";
import { api, askQuestion, deleteDocument, getErrorMessage, listDocuments, uploadDocument } from "./api";

// Replace axios' network layer with a fake, so the real client config
// (base URL, timeouts, body handling) is exercised without a server.
const originalAdapter = api.defaults.adapter;
let requests: InternalAxiosRequestConfig[] = [];

function respondWith(data: unknown, status = 200) {
  const adapter: AxiosAdapter = async (config) => {
    requests.push(config);
    const response = { data, status, statusText: "", headers: {}, config };
    if (status >= 400) throw new AxiosError("Request failed", "ERR_BAD_RESPONSE", config, null, response);
    return response;
  };
  api.defaults.adapter = adapter;
}

afterEach(() => {
  api.defaults.adapter = originalAdapter;
  requests = [];
});

describe("API client", () => {
  it("lists documents from GET /api/documents", async () => {
    respondWith({ document_count: 0, total_chunks: 0, documents: [] });

    await expect(listDocuments()).resolves.toEqual({ document_count: 0, total_chunks: 0, documents: [] });
    expect(requests[0].method).toBe("get");
    expect(requests[0].url).toBe("/api/documents");
    expect(requests[0].baseURL).toBe("http://localhost:8030");
  });

  it("omits unset optional fields when asking a question, so backend defaults apply", async () => {
    respondWith({ answer: "ok" });

    await askQuestion({ question: "What is the notice period?", document_id: null, retrieval_mode: null, rerank: false });

    expect(requests[0].url).toBe("/api/ask");
    expect(JSON.parse(requests[0].data as string)).toEqual({ question: "What is the notice period?", rerank: false });
  });

  it("uploads the file as multipart form data with a long timeout", async () => {
    respondWith({ document_id: "a".repeat(32) });
    const file = new File(["%PDF-1.4"], "handbook.pdf", { type: "application/pdf" });

    await uploadDocument(file);

    const request = requests[0];
    expect(request.url).toBe("/api/documents/upload");
    expect(request.data).toBeInstanceOf(FormData);
    expect((request.data as FormData).get("file")).toBe(file);
    expect(request.timeout).toBeGreaterThan(api.defaults.timeout ?? 0);
  });

  it("deletes a document by id", async () => {
    respondWith({ document_id: "abc", deleted_chunks: 3 });

    await expect(deleteDocument("abc")).resolves.toEqual({ document_id: "abc", deleted_chunks: 3 });
    expect(requests[0].method).toBe("delete");
    expect(requests[0].url).toBe("/api/documents/abc");
  });
});

describe("getErrorMessage", () => {
  const config = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig;
  const withResponse = (status: number, data: unknown) =>
    new AxiosError("failed", "ERR_BAD_RESPONSE", config, null, { data, status, statusText: "", headers: {}, config });

  it("uses FastAPI's string detail", () => {
    expect(getErrorMessage(withResponse(415, { detail: "Only .pdf files are accepted." }))).toBe(
      "Only .pdf files are accepted.",
    );
  });

  it("summarizes FastAPI validation errors", () => {
    const detail = [{ loc: ["body", "question"], msg: "String should have at least 1 character" }];
    expect(getErrorMessage(withResponse(422, { detail }))).toBe("question: String should have at least 1 character");
  });

  it("falls back to the status code when there is no detail", () => {
    expect(getErrorMessage(withResponse(500, "Internal Server Error"))).toBe("Request failed with status 500.");
  });

  it("explains network errors and timeouts", () => {
    expect(getErrorMessage(new AxiosError("Network Error", "ERR_NETWORK", config))).toMatch(/Cannot reach the backend/);
    expect(getErrorMessage(new AxiosError("timeout", "ECONNABORTED", config))).toMatch(/timed out/);
  });

  it("handles non-axios errors", () => {
    expect(getErrorMessage(new Error("boom"))).toBe("boom");
    expect(getErrorMessage("weird")).toBe("Something went wrong.");
  });
});
