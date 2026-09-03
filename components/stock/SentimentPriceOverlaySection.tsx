"use client";

import { useEffect, useState } from "react";
import { SentimentPriceOverlayChart } from "@/components/blog/SentimentPriceOverlayChart";
import type { SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";

export function SentimentPriceOverlaySection({ symbol }: { symbol: string }) {
  const [data, setData] = useState<SentimentPricePoint[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/sentiment-history/${symbol}`)
      .then((res) => res.json())
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch(() => {
        if (!cancelled) setData([]);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  const hasEnoughData = data && data.filter((d) => d.averageSentiment != null).length >= 2;
  if (!hasEnoughData) return null;

  return (
    <section className="mb-8 rounded-xl border border-border bg-card p-4">
      <h2 className="mb-3 text-sm font-medium text-muted-foreground">
        Sentiment vs. Price (90 days)
      </h2>
      <SentimentPriceOverlayChart data={data} />
    </section>
  );
}
