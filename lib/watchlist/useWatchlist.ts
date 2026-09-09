"use client";

import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { useSession } from "@/lib/auth/useSession";
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

interface ServerWatchlistEntry {
  symbol: string;
  companyName: string;
  addedAt: string;
}

/**
 * Dual-mode: logged out -> localStorage (unchanged behavior). Logged in ->
 * server-backed via /api/watchlist, so the watchlist follows the account
 * across devices. Prices are always tracked client-side only, in either mode.
 */
export function useWatchlist() {
  const { loggedIn, isLoading: sessionLoading } = useSession();
  const localEntries = useSyncExternalStore(
    subscribeToWatchlist,
    readWatchlist,
    getServerSnapshot
  );
  const [serverEntries, setServerEntries] = useState<WatchlistEntry[] | null>(null);

  useEffect(() => {
    // When logged out, `entries` below falls back to localEntries regardless
    // of serverEntries, so there's nothing to reset here.
    if (!loggedIn) return;
    let cancelled = false;
    fetch("/api/watchlist")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data) return;
        setServerEntries(
          data.entries.map((e: ServerWatchlistEntry) => ({
            symbol: e.symbol,
            companyName: e.companyName,
            addedAt: e.addedAt,
            lastPrice: null,
            lastChangePercent: null,
          }))
        );
      })
      .catch(() => {
        if (!cancelled) setServerEntries([]);
      });
    return () => {
      cancelled = true;
    };
  }, [loggedIn]);

  const entries = useMemo(
    () => (loggedIn ? serverEntries ?? [] : localEntries),
    [loggedIn, serverEntries, localEntries]
  );

  const add = useCallback(
    (entry: Omit<WatchlistEntry, "addedAt">) => {
      if (loggedIn) {
        setServerEntries((prev) => {
          const current = prev ?? [];
          if (current.some((e) => e.symbol === entry.symbol)) return current;
          return [...current, { ...entry, addedAt: new Date().toISOString() }];
        });
        fetch("/api/watchlist", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol: entry.symbol, companyName: entry.companyName }),
        }).catch(() => {});
      } else {
        addToWatchlist(entry);
      }
    },
    [loggedIn]
  );

  const remove = useCallback(
    (symbol: string) => {
      if (loggedIn) {
        setServerEntries((prev) => (prev ?? []).filter((e) => e.symbol !== symbol));
        fetch(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: "DELETE" }).catch(() => {});
      } else {
        removeFromWatchlist(symbol);
      }
    },
    [loggedIn]
  );

  const has = useCallback((symbol: string) => entries.some((e) => e.symbol === symbol), [entries]);

  const updatePrices = useCallback(
    (prices: Map<string, { price: number | null; changePercent: number | null }>) => {
      if (!loggedIn) return; // logged-out path still uses updateWatchlistPrices(storage) directly
      setServerEntries((prev) =>
        (prev ?? []).map((entry) => {
          const update = prices.get(entry.symbol);
          if (!update) return entry;
          return {
            ...entry,
            lastPrice: update.price ?? entry.lastPrice,
            lastChangePercent: update.changePercent ?? entry.lastChangePercent,
          };
        })
      );
    },
    [loggedIn]
  );

  return { entries, add, remove, has, updatePrices, loggedIn, isLoading: sessionLoading };
}
