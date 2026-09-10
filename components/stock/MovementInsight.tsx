"use client";

import { useEffect, useState } from "react";

interface MovementInsightProps {
  symbol: string;
  companyName: string;
  changePercent: number | null;
  /**
   * Insight already generated server-side, so the initial SSR HTML —
   * what crawlers and AI bots see — already contains the explanation
   * sentence instead of a "loading" placeholder.
   */
  initialInsight?: InsightResult | null;
}

interface InsightResult {
  summary: string;
  source: string;
}

const THRESHOLD_PERCENT = 3;

type FetchState =
  | { status: "loading" }
  | { status: "success"; insight: InsightResult }
  | { status: "error" };

export function MovementInsight({
  symbol,
  companyName,
  changePercent,
  initialInsight,
}: MovementInsightProps) {
  // A single discriminated status keeps loading/success/error mutually
  // exclusive, so a success can never leave a stale error flag set.
  const [state, setState] = useState<FetchState>(
    initialInsight ? { status: "success", insight: initialInsight } : { status: "loading" }
  );
  const [retryCount, setRetryCount] = useState(0);

  const qualifies = changePercent != null && Math.abs(changePercent) >= THRESHOLD_PERCENT;

  useEffect(() => {
    if (!qualifies) return;
    if (initialInsight && retryCount === 0) return;

    let cancelled = false;
    setState({ status: "loading" });

    fetch("/api/movement-insight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, companyName, changePercent }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return;
        const result = data.movement_insight ?? null;
        setState(result ? { status: "success", insight: result } : { status: "error" });
      })
      .catch(() => {
        if (!cancelled) setState({ status: "error" });
      });

    return () => {
      cancelled = true;
    };
  }, [symbol, companyName, changePercent, retryCount]);

  if (!qualifies) return null;

  return (
    <section className="mb-8 rounded-xl border border-primary/30 bg-primary/5 p-4 shadow-sm">
      <h2 className="mb-2 text-sm font-medium text-primary">Why is it moving?</h2>
      {state.status === "loading" ? (
        <p className="text-sm text-muted-foreground">Analyzing recent headlines…</p>
      ) : state.status === "error" ? (
        <div className="flex items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            We couldn&apos;t generate an explanation for this move right now.
          </p>
          <button
            type="button"
            onClick={() => setRetryCount((n) => n + 1)}
            className="shrink-0 rounded-md border border-primary/40 px-3 py-1 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
          >
            Retry
          </button>
        </div>
      ) : (
        <p className="text-base leading-relaxed">{state.insight.summary}</p>
      )}
    </section>
  );
}
