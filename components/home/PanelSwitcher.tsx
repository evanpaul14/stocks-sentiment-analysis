"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useWatchlist } from "@/lib/watchlist/useWatchlist";
import type { TrendingItem } from "@/lib/trending/getTrendingSourceData";

type Panel = "watchlist" | "trending";

export function PanelSwitcher() {
  const [panel, setPanel] = useState<Panel>("trending");
  const { entries } = useWatchlist();
  const [trending, setTrending] = useState<TrendingItem[]>([]);

  useEffect(() => {
    if (panel !== "trending") return;
    fetch("/api/trending/stocktwits")
      .then((res) => res.json())
      .then((data) => setTrending(Array.isArray(data) ? data.slice(0, 5) : []))
      .catch(() => setTrending([]));
  }, [panel]);

  return (
    <div className="w-full max-w-md rounded-xl border border-border bg-card/60 p-4 backdrop-blur">
      <div className="mb-3 flex gap-1 rounded-lg bg-muted p-1 text-sm">
        <button
          type="button"
          onClick={() => setPanel("trending")}
          className={`flex-1 rounded-md py-1.5 transition-colors ${
            panel === "trending" ? "bg-background shadow-sm" : "text-muted-foreground"
          }`}
        >
          Trending
        </button>
        <button
          type="button"
          onClick={() => setPanel("watchlist")}
          className={`flex-1 rounded-md py-1.5 transition-colors ${
            panel === "watchlist" ? "bg-background shadow-sm" : "text-muted-foreground"
          }`}
        >
          Watchlist
        </button>
      </div>

      {panel === "trending" ? (
        <ul className="space-y-1">
          {trending.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted-foreground">Loading…</p>
          ) : (
            trending.map((item) => <TeaserRow key={item.symbol} symbol={item.symbol} label={item.companyName} price={item.price} changePercent={item.changePercent} />)
          )}
        </ul>
      ) : (
        <ul className="space-y-1">
          {entries.length === 0 ? (
            <p className="py-4 text-center text-xs text-muted-foreground">
              No stocks in your watchlist yet.
            </p>
          ) : (
            entries
              .slice(0, 5)
              .map((entry) => (
                <TeaserRow
                  key={entry.symbol}
                  symbol={entry.symbol}
                  label={entry.companyName}
                  price={entry.lastPrice}
                  changePercent={entry.lastChangePercent}
                />
              ))
          )}
        </ul>
      )}
    </div>
  );
}

function TeaserRow({
  symbol,
  label,
  price,
  changePercent,
}: {
  symbol: string;
  label: string;
  price: number | null;
  changePercent: number | null;
}) {
  const isUp = (changePercent ?? 0) >= 0;
  return (
    <li>
      <Link
        href={`/stock/${symbol}`}
        className="flex items-center justify-between gap-3 rounded-lg px-2 py-1.5 text-sm hover:bg-muted"
      >
        <span className="min-w-0 flex-1 truncate text-muted-foreground">
          <span className="font-medium text-foreground">{symbol}</span> {label}
        </span>
        <span className="shrink-0 whitespace-nowrap tabular-nums">
          {price != null && (
            <>
              <span>${price.toFixed(2)}</span>{" "}
              {changePercent != null && (
                <span className={isUp ? "text-[var(--color-chart-1)]" : "text-destructive"}>
                  {isUp ? "+" : ""}
                  {changePercent.toFixed(2)}%
                </span>
              )}
            </>
          )}
        </span>
      </Link>
    </li>
  );
}
