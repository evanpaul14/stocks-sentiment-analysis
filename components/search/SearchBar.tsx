"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useSearchHistory } from "@/lib/searchHistory/useSearchHistory";

interface SearchBarProps {
  showSuggestions?: boolean;
  className?: string;
}

export function SearchBar({ showSuggestions = false, className = "max-w-md" }: SearchBarProps) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const { history, record } = useSearchHistory();
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const instanceId = useId();
  const inputId = `stock-search-input-${instanceId}`;
  const listboxId = `stock-search-suggestions-${instanceId}`;
  const optionId = (index: number) => `stock-search-option-${instanceId}-${index}`;

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
      record(trimmed);
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
    <div ref={containerRef} className={`w-full ${className}`}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(query);
        }}
        className="flex gap-2"
      >
        <div className="relative min-w-0 flex-1">
          <label htmlFor={inputId} className="sr-only">
            Search a ticker or company
          </label>
          <input
            id={inputId}
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
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none transition-shadow focus:border-ring focus:ring-3 focus:ring-ring/50"
            autoComplete="off"
            role={showSuggestions ? "combobox" : undefined}
            aria-expanded={showSuggestions ? isOpen && filteredHistory.length > 0 : undefined}
            aria-controls={showSuggestions ? listboxId : undefined}
            aria-autocomplete={showSuggestions ? "list" : undefined}
            aria-activedescendant={
              showSuggestions && highlightedIndex >= 0 ? optionId(highlightedIndex) : undefined
            }
          />

          {showSuggestions && isOpen && filteredHistory.length > 0 && (
            <ul
              id={listboxId}
              role="listbox"
              className="animate-fade-in-down absolute z-10 mt-1 w-full overflow-hidden rounded-lg border border-border bg-popover shadow-lg"
            >
              {filteredHistory.map((entry, index) => (
                <li key={entry} role="presentation">
                  <button
                    type="button"
                    id={optionId(index)}
                    role="option"
                    aria-selected={index === highlightedIndex}
                    className={`block w-full px-3 py-2 text-left text-sm transition-colors duration-100 ${
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
        <Button type="submit">Search</Button>
      </form>
    </div>
  );
}
