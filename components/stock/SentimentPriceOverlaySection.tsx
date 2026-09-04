"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import type { SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";

const SentimentPriceOverlayChart = dynamic(
  () =>
    import("@/components/blog/SentimentPriceOverlayChart").then(
      (m) => m.SentimentPriceOverlayChart
    ),
  { ssr: false }
);

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
