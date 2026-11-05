import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AxiosError } from "axios";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatProvider } from "../context/ChatContext";
import { DocumentsProvider } from "../context/DocumentsContext";
import * as api from "../services/api";
import { CONVERSATION_ID, DOC_ID, answered, documentList, emptyDocumentList, notFound, savedConversation } from "../test/fixtures";
import Chat from "./Chat";

vi.mock("../services/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../services/api")>()),
  listDocuments: vi.fn(),
  askQuestion: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  deleteConversation: vi.fn(),
}));

const listDocuments = vi.mocked(api.listDocuments);
const askQuestion = vi.mocked(api.askQuestion);
const listConversations = vi.mocked(api.listConversations);
const getConversation = vi.mocked(api.getConversation);
const deleteConversation = vi.mocked(api.deleteConversation);

const savedSummary = {
  id: CONVERSATION_ID,
  title: savedConversation.title,
  document_id: null,
  message_count: 2,
  created_at: savedConversation.created_at,
  updated_at: savedConversation.updated_at,
};

function renderChat() {
  return render(
    <DocumentsProvider>
      <ChatProvider>
        <MemoryRouter>
          <Chat />
        </MemoryRouter>
      </ChatProvider>
    </DocumentsProvider>,
  );
}

async function ask(question: string) {
  await userEvent.type(screen.getByLabelText("Your question"), `${question}{Enter}`);
}

beforeEach(() => {
  vi.clearAllMocks();
  listDocuments.mockResolvedValue(documentList);
  listConversations.mockResolvedValue({ total: 0, conversations: [] });
});

describe("Chat page", () => {
  it("shows the answer with its source filename, page and excerpt", async () => {
    askQuestion.mockResolvedValue(answered);
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("What is the notice period?");

    expect(await screen.findByText(/two months of written notice/)).toBeInTheDocument();
    expect(screen.getByText("What is the notice period?")).toBeInTheDocument();
    expect(screen.getByText("Page 6")).toBeInTheDocument();
    expect(screen.getByText("After the first year, the notice period is two months.")).toBeInTheDocument();
    expect(askQuestion).toHaveBeenCalledWith({
      question: "What is the notice period?",
      conversation_id: null,
      document_id: null,
      retrieval_mode: null,
      rerank: null,
    });
  });

  it("sends the selected document and retrieval mode", async () => {
    askQuestion.mockResolvedValue(answered);
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await userEvent.selectOptions(screen.getByLabelText("Search in"), DOC_ID);
    await userEvent.selectOptions(screen.getByLabelText("Retrieval"), "dense");
    await ask("Notice?");

    expect(askQuestion).toHaveBeenCalledWith(expect.objectContaining({ document_id: DOC_ID, retrieval_mode: "dense" }));
  });

  it("shows a loading indicator while waiting for the answer", async () => {
    let resolve: (value: typeof answered) => void = () => {};
    askQuestion.mockReturnValue(new Promise((r) => (resolve = r)));
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("Notice?");

    expect(screen.getByText(/generating an answer/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send question" })).toBeDisabled();
    resolve(answered);
    expect(await screen.findByText(/two months of written notice/)).toBeInTheDocument();
    expect(screen.queryByText(/generating an answer/)).not.toBeInTheDocument();
  });

  it("explains when the documents contain no supporting evidence", async () => {
    askQuestion.mockResolvedValue(notFound);
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("What is the CEO's favourite colour?");

    expect(await screen.findByText("No supporting evidence found")).toBeInTheDocument();
    expect(screen.getByText(notFound.answer)).toBeInTheDocument();
    expect(screen.queryByText(/Sources/)).not.toBeInTheDocument();
  });

  it("shows an error message when the backend cannot be reached", async () => {
    askQuestion.mockRejectedValue(new AxiosError("Network Error", "ERR_NETWORK"));
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("Notice?");

    expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the backend/);
  });

  it("asks the user to upload a document when none are indexed", async () => {
    listDocuments.mockResolvedValue(emptyDocumentList);
    renderChat();

    expect(await screen.findByText(/No documents are indexed yet/)).toBeInTheDocument();
    expect(screen.getByLabelText("Your question")).toBeDisabled();
  });

  it("only offers indexed documents", async () => {
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    expect(screen.queryByRole("option", { name: "broken.pdf" })).not.toBeInTheDocument();
  });

  it("continues the saved conversation on follow-up questions", async () => {
    askQuestion.mockResolvedValue(answered);
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("What is the notice period?");
    await screen.findByText(/two months of written notice/);
    await ask("And during probation?");

    await vi.waitFor(() => expect(askQuestion).toHaveBeenCalledTimes(2));
    expect(askQuestion.mock.calls[0][0].conversation_id).toBeNull();
    expect(askQuestion.mock.calls[1][0].conversation_id).toBe(CONVERSATION_ID);
    // The history list is refreshed after a saved answer.
    expect(listConversations.mock.calls.length).toBeGreaterThan(1);
  });

  it("opens a saved conversation with its answers and sources", async () => {
    listConversations.mockResolvedValue({ total: 1, conversations: [savedSummary] });
    getConversation.mockResolvedValue(savedConversation);
    askQuestion.mockResolvedValue(answered);
    renderChat();

    await userEvent.click(await screen.findByRole("button", { name: /^What is the notice period\?/ }));

    expect(await screen.findByText(/two months of written notice/)).toBeInTheDocument();
    expect(screen.getByText("Page 6")).toBeInTheDocument();
    expect(getConversation).toHaveBeenCalledWith(CONVERSATION_ID);

    await ask("Anything else?");
    expect(askQuestion).toHaveBeenCalledWith(expect.objectContaining({ conversation_id: CONVERSATION_ID }));
  });

  it("starts a new conversation with New chat", async () => {
    askQuestion.mockResolvedValueOnce(answered).mockResolvedValueOnce(notFound);
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });
    await ask("What is the notice period?");
    await screen.findByText(/two months of written notice/);

    await userEvent.click(screen.getAllByRole("button", { name: /New chat/ })[0]);
    await ask("A new topic?");

    expect(screen.queryByText(/two months of written notice/)).not.toBeInTheDocument();
    await vi.waitFor(() => expect(askQuestion).toHaveBeenCalledTimes(2));
    expect(askQuestion.mock.calls[1][0].conversation_id).toBeNull();
  });

  it("deletes a saved conversation after confirmation", async () => {
    listConversations.mockResolvedValue({ total: 1, conversations: [savedSummary] });
    deleteConversation.mockResolvedValue();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderChat();

    await userEvent.click(await screen.findByRole("button", { name: `Delete ${savedSummary.title}` }));

    expect(deleteConversation).toHaveBeenCalledWith(CONVERSATION_ID);
  });

  it("warns when an answer could not be saved to history", async () => {
    askQuestion.mockResolvedValue({ ...answered, history_saved: false, conversation_id: null, message_id: null });
    renderChat();
    await screen.findByRole("option", { name: "handbook.pdf" });

    await ask("What is the notice period?");

    expect(await screen.findByText("Not saved to history")).toBeInTheDocument();
  });
});
