"use client";

import { useEffect, useState } from "react";

interface LivePriceProps {
  symbol: string;
  initialPrice: number | null;
  initialChangePercent: number | null;
  /** Extended-hours move since the regular close, shown under the headline. */
  postMarketChange?: number | null;
  postMarketChangePercent?: number | null;
}

const POLL_INTERVAL_MS = 15_000;

export function LivePrice({
  symbol,
  initialPrice,
  initialChangePercent,
  postMarketChange,
  postMarketChangePercent,
}: LivePriceProps) {
  const [price, setPrice] = useState(initialPrice);
  const [changePercent, setChangePercent] = useState(initialChangePercent);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const response = await fetch(`/api/quote/${symbol}`);
        if (!response.ok) return;
        const data = await response.json();
        if (cancelled) return;
        if (typeof data.price === "number") setPrice(data.price);
        if (typeof data.changePercent === "number")
          setChangePercent(data.changePercent);
      } catch {
        // silent: keep showing the last known price
      }
    }

    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [symbol]);

  const isUp = (changePercent ?? 0) >= 0;

  return (
    <div>
      <div className="flex items-baseline gap-3">
        <span className="text-3xl font-semibold tabular-nums">
          {price != null ? `$${price.toFixed(2)}` : "—"}
        </span>
        {changePercent != null && (
          <span
            className={`text-sm font-medium tabular-nums ${
              isUp ? "text-[var(--color-chart-1)]" : "text-destructive"
            }`}
          >
            {isUp ? "+" : ""}
            {changePercent.toFixed(2)}%
            <span className="ml-1 font-normal text-muted-foreground">
              Today
            </span>
          </span>
        )}
      </div>
      {postMarketChange != null && postMarketChangePercent != null && (
        <div
          className={`mt-1 text-sm tabular-nums ${
            postMarketChange >= 0
              ? "text-[var(--color-chart-1)]"
              : "text-destructive"
          }`}
        >
          {postMarketChange >= 0 ? "+" : "−"}$
          {Math.abs(postMarketChange).toFixed(2)} (
          {postMarketChange >= 0 ? "+" : "−"}
          {Math.abs(postMarketChangePercent).toFixed(2)}%)
          <span className="ml-1 text-muted-foreground">After-hours</span>
        </div>
      )}
    </div>
  );
}
