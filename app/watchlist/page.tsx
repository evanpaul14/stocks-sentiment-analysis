"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/lib/watchlist/useWatchlist";
import { updateWatchlistPrices } from "@/lib/watchlist/storage";

const REFRESH_INTERVAL_MS = 30_000;

export default function WatchlistPage() {
  const { entries, remove, updatePrices, loggedIn } = useWatchlist();
  const symbolsRef = useRef<string[]>([]);
  const listRef = useRef<HTMLUListElement>(null);
  const emptyStateButtonRef = useRef<HTMLButtonElement>(null);
  // Index of the row whose remove button was just clicked. Removing a row
  // unmounts the focused button, so once the list re-renders, focus moves to
  // the next row's remove button (or the empty-state button).
  const focusAfterRemoveRef = useRef<number | null>(null);

  useEffect(() => {
    symbolsRef.current = entries.map((e) => e.symbol);
  }, [entries]);

  useEffect(() => {
    const index = focusAfterRemoveRef.current;
    if (index === null) return;
    focusAfterRemoveRef.current = null;
    const buttons = listRef.current?.querySelectorAll<HTMLButtonElement>("[data-remove-button]");
    if (buttons && buttons.length > 0) {
      buttons[Math.min(index, buttons.length - 1)].focus();
    } else {
      emptyStateButtonRef.current?.focus();
    }
  }, [entries]);

  useEffect(() => {
    async function refreshPrices() {
      const symbols = symbolsRef.current;
      if (symbols.length === 0) return;
      try {
        const response = await fetch("/api/trending/prices", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbols }),
        });
        if (!response.ok) return;
        const snapshots: Array<{
          symbol: string;
          price: number | null;
          changePercent: number | null;
        }> = await response.json();
        const priceMap = new Map(snapshots.map((s) => [s.symbol, s]));
        if (loggedIn) {
          updatePrices(priceMap);
        } else {
          updateWatchlistPrices(priceMap);
        }
      } catch {
        // silent: keep showing last known prices
      }
    }

    refreshPrices();
    const interval = setInterval(refreshPrices, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [loggedIn, updatePrices]);

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Watchlist</h1>

      {entries.length === 0 ? (
        <div className="flex flex-col items-start gap-3">
          <p className="text-sm text-muted-foreground">
            Your watchlist is empty. Search a stock and add it from its page.
          </p>
          <Button
            ref={emptyStateButtonRef}
            type="button"
            onClick={() => window.dispatchEvent(new Event("open-search"))}
          >
            Add to watchlist
          </Button>
        </div>
      ) : (
        <ul ref={listRef} className="space-y-2">
          {entries.map((entry, index) => {
            const isUp = (entry.lastChangePercent ?? 0) >= 0;
            return (
              <li
                key={entry.symbol}
                className="flex items-center justify-between gap-3 rounded-none border border-border p-3 transition-colors duration-150 hover:border-foreground/20 hover:bg-muted/50"
              >
                <Link
                  href={`/stock/${entry.symbol}`}
                  className="min-w-0 flex-1"
                >
                  <p className="font-medium">{entry.symbol}</p>
                  <p className="truncate text-xs text-muted-foreground">
                    {entry.companyName}
                  </p>
                </Link>
                <div className="text-right text-sm tabular-nums">
                  <p>{entry.lastPrice != null ? `$${entry.lastPrice.toFixed(2)}` : "—"}</p>
                  {entry.lastChangePercent != null && (
                    <p className={isUp ? "text-[var(--color-chart-1)]" : "text-destructive-text"}>
                      {isUp ? "+" : ""}
                      {entry.lastChangePercent.toFixed(2)}%
                    </p>
                  )}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  aria-label={`Remove ${entry.symbol} from watchlist`}
                  data-remove-button
                  onClick={() => {
                    focusAfterRemoveRef.current = index;
                    remove(entry.symbol);
                  }}
                >
                  <X />
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </main>
  );
}
