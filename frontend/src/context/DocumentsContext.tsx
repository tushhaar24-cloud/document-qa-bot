// Shared document list (from PostgreSQL via the backend). Several pages need it
// (dashboard, documents, chat), so it is fetched once and shared here.

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getErrorMessage, listDocuments } from "../services/api";
import type { DocumentListResponse } from "../types";

interface DocumentsState {
  data: DocumentListResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

const DocumentsContext = createContext<DocumentsState | null>(null);

export function DocumentsProvider({ children }: { children: ReactNode }) {
  const [data, setData] = useState<DocumentListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setData(await listDocuments());
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const value = useMemo(() => ({ data, loading, error, refresh }), [data, loading, error, refresh]);
  return <DocumentsContext.Provider value={value}>{children}</DocumentsContext.Provider>;
}

export function useDocuments(): DocumentsState {
  const context = useContext(DocumentsContext);
  if (!context) throw new Error("useDocuments must be used inside <DocumentsProvider>");
  return context;
}
