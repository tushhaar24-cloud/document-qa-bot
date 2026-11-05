import { ChartColumn, FileText, LayoutDashboard, MessageSquare, type LucideIcon } from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  description: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", description: "Overview of your indexed documents and evaluation results", icon: LayoutDashboard },
  { to: "/documents", label: "Documents", description: "Upload, review and remove PDF documents", icon: FileText },
  { to: "/chat", label: "AI Chat", description: "Ask questions and get answers with page-level citations", icon: MessageSquare },
  { to: "/evaluation", label: "Evaluation", description: "Ragas and retrieval metrics from saved evaluation runs", icon: ChartColumn },
];
