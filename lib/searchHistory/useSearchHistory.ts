"use client";

import { useCallback, useEffect, useState, useSyncExternalStore } from "react";
import { useSession } from "@/lib/auth/useSession";
import { readHistory, recordSearch, subscribeToHistory } from "./storage";

function getServerSnapshot(): string[] {
  return [];
}

/**
 * Dual-mode, mirroring useWatchlist: logged out -> localStorage (unchanged
 * behavior). Logged in -> server-backed via /api/search-history.
 */
export function useSearchHistory() {
  const { loggedIn } = useSession();
  const localHistory = useSyncExternalStore(subscribeToHistory, readHistory, getServerSnapshot);
  const [serverHistory, setServerHistory] = useState<string[] | null>(null);

  useEffect(() => {
    // When logged out, `history` below falls back to localHistory regardless
    // of serverHistory, so there's nothing to reset here.
    if (!loggedIn) return;
    let cancelled = false;
    fetch("/api/search-history")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setServerHistory(data.queries);
      })
      .catch(() => {
        if (!cancelled) setServerHistory([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loggedIn]);

  const history = loggedIn ? serverHistory ?? [] : localHistory;

  const record = useCallback(
    (term: string) => {
      const trimmed = term.trim();
      if (!trimmed) return;

      if (loggedIn) {
        setServerHistory((prev) =>
          [trimmed, ...(prev ?? []).filter((h) => h.toLowerCase() !== trimmed.toLowerCase())].slice(0, 8)
        );
        fetch("/api/search-history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: trimmed }),
        }).catch(() => {});
      } else {
        recordSearch(trimmed);
      }
    },
    [loggedIn]
  );

  return { history, record };
}
