"use client";

import { useEffect, useState } from "react";

interface MovementInsightProps {
  symbol: string;
  companyName: string;
  changePercent: number | null;
}

interface InsightResult {
  summary: string;
  source: string;
}

const THRESHOLD_PERCENT = 3;

export function MovementInsight({ symbol, companyName, changePercent }: MovementInsightProps) {
  const [insight, setInsight] = useState<InsightResult | null>(null);
  // Starts true: whenever this effect actually runs (qualifying move), it's
  // fetching immediately, so there's no "not loading yet" state to model.
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (changePercent == null || Math.abs(changePercent) < THRESHOLD_PERCENT) return;

    let cancelled = false;

    fetch("/api/movement-insight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbol, companyName, changePercent }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (!cancelled) setInsight(data.movement_insight ?? null);
      })
      .catch(() => {
        // silent: this is a nice-to-have, page still works without it
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [symbol, companyName, changePercent]);

  if (changePercent == null || Math.abs(changePercent) < THRESHOLD_PERCENT) return null;
  if (!loading && !insight) return null;

  return (
    <section className="mb-8 rounded-xl border border-border bg-card p-4">
      <h2 className="mb-2 text-sm font-medium text-muted-foreground">Why is it moving?</h2>
      {loading && !insight ? (
        <p className="text-sm text-muted-foreground">Analyzing recent headlines…</p>
      ) : (
        <p className="text-sm leading-relaxed">{insight?.summary}</p>
      )}
    </section>
  );
}
