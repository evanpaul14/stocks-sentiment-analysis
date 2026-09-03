"use client";

import { useCallback, useSyncExternalStore } from "react";
import {
  addToWatchlist,
  readWatchlist,
  removeFromWatchlist,
  subscribeToWatchlist,
  type WatchlistEntry,
} from "./storage";

function getServerSnapshot(): WatchlistEntry[] {
  return [];
}

export function useWatchlist() {
  const entries = useSyncExternalStore(
    subscribeToWatchlist,
    readWatchlist,
    getServerSnapshot
  );

  const add = useCallback((entry: Omit<WatchlistEntry, "addedAt">) => {
    addToWatchlist(entry);
  }, []);

  const remove = useCallback((symbol: string) => {
    removeFromWatchlist(symbol);
  }, []);

  const has = useCallback(
    (symbol: string) => entries.some((e) => e.symbol === symbol),
    [entries]
  );

  return { entries, add, remove, has };
}
