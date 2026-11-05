import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AxiosError, AxiosHeaders, type InternalAxiosRequestConfig } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DocumentsProvider } from "../../context/DocumentsContext";
import * as api from "../../services/api";
import { DOC_ID, documentList, emptyDocumentList, uploadResult } from "../../test/fixtures";
import DocumentUpload from "./DocumentUpload";

// Mock only the network functions; getErrorMessage stays real.
vi.mock("../../services/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../services/api")>()),
  listDocuments: vi.fn(),
  uploadDocument: vi.fn(),
}));

const listDocuments = vi.mocked(api.listDocuments);
const uploadDocument = vi.mocked(api.uploadDocument);

function renderUpload() {
  return render(
    <DocumentsProvider>
      <DocumentUpload />
    </DocumentsProvider>,
  );
}

const pdf = (name = "handbook.pdf") => new File(["%PDF-1.4"], name, { type: "application/pdf" });

beforeEach(() => {
  vi.clearAllMocks();
  listDocuments.mockResolvedValue(emptyDocumentList);
});

describe("DocumentUpload", () => {
  it("uploads a PDF, shows its document ID and refreshes the document list", async () => {
    uploadDocument.mockImplementation(async (_file, onProgress) => {
      onProgress?.(100);
      return uploadResult;
    });
    listDocuments.mockResolvedValueOnce(emptyDocumentList).mockResolvedValueOnce(documentList);
    renderUpload();

    await userEvent.upload(screen.getByLabelText(/drop pdf files here/i), pdf());

    expect(await screen.findByText("Indexed")).toBeInTheDocument();
    expect(screen.getByText(DOC_ID)).toBeInTheDocument();
    expect(uploadDocument).toHaveBeenCalledWith(expect.any(File), expect.any(Function));
    await waitFor(() => expect(listDocuments).toHaveBeenCalledTimes(2));
  });

  it("uploads several files one after another", async () => {
    uploadDocument.mockResolvedValue(uploadResult);
    renderUpload();

    await userEvent.upload(screen.getByLabelText(/drop pdf files here/i), [pdf("a.pdf"), pdf("b.pdf")]);

    await waitFor(() => expect(screen.getAllByText("Indexed")).toHaveLength(2));
    expect(uploadDocument.mock.calls.map(([file]) => file.name)).toEqual(["a.pdf", "b.pdf"]);
  });

  it("rejects non-PDF files without calling the backend", async () => {
    renderUpload();

    const input = screen.getByLabelText(/drop pdf files here/i);
    await userEvent.upload(input, new File(["hello"], "notes.txt", { type: "text/plain" }), { applyAccept: false });

    expect(await screen.findByText("Only .pdf files are accepted.")).toBeInTheDocument();
    expect(uploadDocument).not.toHaveBeenCalled();
  });

  it("shows the backend's error message when indexing fails", async () => {
    const config = { headers: new AxiosHeaders() } as InternalAxiosRequestConfig;
    uploadDocument.mockRejectedValue(
      new AxiosError("failed", "ERR_BAD_RESPONSE", config, null, {
        data: { detail: "The PDF is password-protected." },
        status: 422,
        statusText: "",
        headers: {},
        config,
      }),
    );
    renderUpload();

    await userEvent.upload(screen.getByLabelText(/drop pdf files here/i), pdf());

    expect(await screen.findByText("The PDF is password-protected.")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });
});
