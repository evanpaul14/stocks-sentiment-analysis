"use client";

import { useMemo, useState, useTransition } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PricePoint } from "@/lib/integrations/yahoo/historical";

interface PriceChartProps {
  symbol: string;
  data: PricePoint[];
}

const RANGES: { label: string; period: string }[] = [
  { label: "Day", period: "1d" },
  { label: "5D", period: "5d" },
  { label: "1M", period: "1mo" },
  { label: "YTD", period: "ytd" },
  { label: "1Y", period: "1y" },
  { label: "5Y", period: "5y" },
];

function formatChange(amount: number, percent: number): string {
  const sign = amount >= 0 ? "+" : "−";
  return `${sign}$${Math.abs(amount).toFixed(2)} (${sign}${Math.abs(percent).toFixed(2)}%)`;
}

export function PriceChart({ symbol, data: initialData }: PriceChartProps) {
  const [period, setPeriod] = useState("1d");
  const [data, setData] = useState(initialData);
  const [isPending, startTransition] = useTransition();

  const change = useMemo(() => {
    if (data.length < 2) return null;
    // The Day range is covered by the headline quote (vs previous close).
    if (period === "1d") return null;
    const first = data[0].price;
    const last = data[data.length - 1].price;
    if (!first) return null;
    return { amount: last - first, percent: ((last - first) / first) * 100 };
  }, [data, period]);

  const isUp = (change?.amount ?? 0) >= 0;
  const color = isUp ? "var(--color-chart-1)" : "var(--color-destructive)";

  function handleRangeChange(nextPeriod: string) {
    if (nextPeriod === period) return;
    setPeriod(nextPeriod);
    startTransition(async () => {
      try {
        const res = await fetch(`/api/historical/${symbol}/${nextPeriod}`);
        if (!res.ok) return;
        const next: PricePoint[] = await res.json();
        setData(next);
      } catch {
        // keep showing the previous range's data on failure
      }
    });
  }

  const rangeLabel = RANGES.find((r) => r.period === period)?.label ?? period;
  // The chart's axes are hidden, so give screen readers the gist in words.
  const chartSummary =
    data.length >= 2
      ? `${symbol} price, ${rangeLabel} range: $${data[0].price.toFixed(2)} on ${data[0].date} to $${data[
          data.length - 1
        ].price.toFixed(2)} on ${data[data.length - 1].date}.`
      : null;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-1">
        <div className={`flex flex-wrap items-baseline gap-x-4 text-sm font-medium tabular-nums ${isPending ? "opacity-60" : ""}`}>
          {change && (
            <span style={{ color: change.amount >= 0 ? "var(--color-chart-1)" : "var(--color-destructive-text)" }}>
              {formatChange(change.amount, change.percent)}
            </span>
          )}
        </div>
        <div role="group" aria-label="Chart range" className="flex gap-1">
        {RANGES.map((range) => (
          <button
            key={range.period}
            type="button"
            aria-pressed={period === range.period}
            onClick={() => handleRangeChange(range.period)}
            className={`rounded-none px-2 py-1 text-xs font-medium transition-colors ${
              period === range.period
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted"
            }`}
          >
            {range.label}
          </button>
        ))}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
          No price data available
        </div>
      ) : (
        <div className={`h-64 w-full ${isPending ? "opacity-60" : ""}`}>
          {chartSummary && <p className="sr-only">{chartSummary}</p>}
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <XAxis dataKey="date" hide />
              <YAxis domain={["auto", "auto"]} hide />
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 0,
                  color: "var(--color-popover-foreground)",
                  fontSize: 12,
                }}
                labelFormatter={(label) => label}
                formatter={(value) => [`$${Number(value).toFixed(2)}`, "Price"]}
              />
              <Line
                type="monotone"
                dataKey="price"
                stroke={color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
