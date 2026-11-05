import { Menu } from "lucide-react";
import { useLocation } from "react-router-dom";
import { NAV_ITEMS } from "./navigation";

export default function Header({ onMenuClick }: { onMenuClick: () => void }) {
  const { pathname } = useLocation();
  const current = NAV_ITEMS.find((item) => item.to === pathname) ?? NAV_ITEMS[0];

  return (
    <header className="sticky top-0 z-20 flex h-16 shrink-0 items-center gap-3 border-b border-slate-200 bg-white px-4 sm:px-6">
      <button
        type="button"
        onClick={onMenuClick}
        className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden"
        aria-label="Open menu"
      >
        <Menu className="h-5 w-5" />
      </button>
      <div className="min-w-0">
        <h1 className="truncate text-base font-semibold text-slate-900">{current.label}</h1>
        <p className="hidden truncate text-xs text-slate-500 sm:block">{current.description}</p>
      </div>
    </header>
  );
}
