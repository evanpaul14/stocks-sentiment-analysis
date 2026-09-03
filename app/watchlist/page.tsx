"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/lib/watchlist/useWatchlist";
import { updateWatchlistPrices } from "@/lib/watchlist/storage";

const REFRESH_INTERVAL_MS = 30_000;

export default function WatchlistPage() {
  const { entries, remove } = useWatchlist();
  const symbolsRef = useRef<string[]>([]);

  useEffect(() => {
    symbolsRef.current = entries.map((e) => e.symbol);
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
        updateWatchlistPrices(
          new Map(snapshots.map((s) => [s.symbol, s]))
        );
      } catch {
        // silent: keep showing last known prices
      }
    }

    refreshPrices();
    const interval = setInterval(refreshPrices, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Watchlist</h1>

      {entries.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Your watchlist is empty. Search a stock and add it from its page.
        </p>
      ) : (
        <ul className="space-y-2">
          {entries.map((entry) => {
            const isUp = (entry.lastChangePercent ?? 0) >= 0;
            return (
              <li
                key={entry.symbol}
                className="flex items-center justify-between gap-3 rounded-lg border border-border p-3 transition-colors duration-150 hover:border-foreground/20 hover:bg-muted/50"
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
                    <p className={isUp ? "text-[var(--color-chart-1)]" : "text-destructive"}>
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
                  onClick={() => remove(entry.symbol)}
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
