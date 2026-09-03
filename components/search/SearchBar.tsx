"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

const HISTORY_KEY = "ssa_recent_searches_v1";
const HISTORY_CHANGED_EVENT = "ssa-search-history-changed";
const MAX_HISTORY = 8;

let cachedRaw: string | null | undefined;
let cachedHistory: string[] = [];

function readHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (raw === cachedRaw) return cachedHistory;

    cachedRaw = raw;
    if (!raw) {
      cachedHistory = [];
    } else {
      const parsed = JSON.parse(raw);
      cachedHistory = Array.isArray(parsed)
        ? parsed.filter((x) => typeof x === "string")
        : [];
    }
    return cachedHistory;
  } catch {
    return [];
  }
}

function writeHistory(entries: string[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
    window.dispatchEvent(new Event(HISTORY_CHANGED_EVENT));
  } catch {
    // localStorage unavailable (private browsing, etc.) — degrade silently
  }
}

function subscribeToHistory(callback: () => void): () => void {
  window.addEventListener(HISTORY_CHANGED_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(HISTORY_CHANGED_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

function getServerHistorySnapshot(): string[] {
  return [];
}

export function SearchBar({ showSuggestions = false }: { showSuggestions?: boolean }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const history = useSyncExternalStore(
    subscribeToHistory,
    readHistory,
    getServerHistorySnapshot
  );
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!showSuggestions) return;
    function handleClickOutside(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showSuggestions]);

  const filteredHistory = showSuggestions
    ? history.filter((h) => h.toLowerCase().includes(query.trim().toLowerCase()))
    : [];

  function submit(value: string) {
    const trimmed = value.trim();
    if (!trimmed) return;

    if (showSuggestions) {
      const next = [trimmed, ...history.filter((h) => h.toLowerCase() !== trimmed.toLowerCase())];
      writeHistory(next);
    }
    setIsOpen(false);
    setHighlightedIndex(-1);

    router.push(`/stock/${encodeURIComponent(trimmed)}`);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!isOpen || filteredHistory.length === 0) {
      if (event.key === "Enter") submit(query);
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlightedIndex((i) => Math.min(i + 1, filteredHistory.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlightedIndex((i) => Math.max(i - 1, -1));
    } else if (event.key === "Enter") {
      event.preventDefault();
      submit(highlightedIndex >= 0 ? filteredHistory[highlightedIndex] : query);
    } else if (event.key === "Escape") {
      setIsOpen(false);
    }
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(query);
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            if (showSuggestions) {
              setIsOpen(true);
              setHighlightedIndex(-1);
            }
          }}
          onFocus={() => showSuggestions && setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder="Search a ticker or company (e.g. AAPL, Apple)"
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
          autoComplete="off"
        />
        <Button type="submit">Search</Button>
      </form>

      {showSuggestions && isOpen && filteredHistory.length > 0 && (
        <ul className="absolute z-10 mt-1 w-full overflow-hidden rounded-lg border border-border bg-popover shadow-lg">
          {filteredHistory.map((entry, index) => (
            <li key={entry}>
              <button
                type="button"
                className={`block w-full px-3 py-2 text-left text-sm ${
                  index === highlightedIndex ? "bg-muted" : "hover:bg-muted"
                }`}
                onMouseDown={(e) => {
                  e.preventDefault();
                  submit(entry);
                }}
              >
                {entry}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
