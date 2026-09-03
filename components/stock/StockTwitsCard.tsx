"use client";

import { useEffect, useState } from "react";
import type { FeedMessage } from "@/lib/integrations/stocktwits/media";
import type { StockTwitsSentimentResult } from "@/lib/integrations/stocktwits/sentiment";

interface StockTwitsCardProps {
  symbol: string;
}

interface StockTwitsCardData {
  sentiment: StockTwitsSentimentResult;
  feed: FeedMessage[];
}

export function StockTwitsCard({ symbol }: StockTwitsCardProps) {
  const [data, setData] = useState<StockTwitsCardData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    fetch(`/api/stocktwits/${symbol}/sentiment`)
      .then((res) => res.json())
      .then((json) => {
        if (!cancelled && json.sentiment) setData(json);
      })
      .catch(() => {
        // silent: this card is a nice-to-have
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (loading) {
    return (
      <section className="mb-8 rounded-xl border border-border bg-card p-4">
        <p className="text-sm text-muted-foreground">Loading StockTwits sentiment…</p>
      </section>
    );
  }

  if (!data) return null;

  const { sentiment, feed } = data;

  return (
    <section className="mb-8 rounded-xl border border-border bg-card p-4">
      <h2 className="mb-3 text-sm font-medium text-muted-foreground">StockTwits Sentiment</h2>

      <div className="mb-4">
        <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-[var(--color-chart-1)]"
            style={{ width: `${sentiment.bullishPercent}%` }}
          />
          <div
            className="h-full bg-destructive"
            style={{ width: `${sentiment.bearishPercent}%` }}
          />
        </div>
        <div className="mt-1 flex justify-between text-xs">
          <span className="text-[var(--color-chart-1)]">
            Bullish {sentiment.bullishPercent}%
          </span>
          <span className="text-destructive">Bearish {sentiment.bearishPercent}%</span>
        </div>
      </div>

      <ul className="max-h-80 space-y-2 overflow-y-auto">
        {feed.slice(0, 15).map((message) => (
          <li key={message.id} className="rounded-lg border border-border p-2 text-sm">
            <div className="flex items-center justify-between">
              <a
                href={message.profileUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium hover:underline"
              >
                @{message.username}
              </a>
              {message.sentiment && (
                <span
                  className={`text-xs capitalize ${
                    message.sentiment === "bullish"
                      ? "text-[var(--color-chart-1)]"
                      : "text-destructive"
                  }`}
                >
                  {message.sentiment}
                </span>
              )}
            </div>
            <p className="mt-0.5 text-muted-foreground">{message.body}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}
