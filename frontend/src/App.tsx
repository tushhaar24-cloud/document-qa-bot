import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/layout/Layout";
import { LoadingState } from "./components/ui/States";
import { ChatProvider } from "./context/ChatContext";
import { DocumentsProvider } from "./context/DocumentsContext";
import Chat from "./pages/Chat";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";

// Recharts is large and only this page uses it, so it is downloaded on first visit.
const Evaluation = lazy(() => import("./pages/Evaluation"));

export default function App() {
  return (
    // Providers sit above the router, so document data and chat history survive page changes.
    <DocumentsProvider>
      <ChatProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="documents" element={<Documents />} />
              <Route path="chat" element={<Chat />} />
              <Route
                path="evaluation"
                element={
                  <Suspense fallback={<LoadingState label="Loading evaluation…" />}>
                    <Evaluation />
                  </Suspense>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ChatProvider>
    </DocumentsProvider>
  );
}
