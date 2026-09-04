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

export function PriceChart({ symbol, data: initialData }: PriceChartProps) {
  const [period, setPeriod] = useState("1d");
  const [data, setData] = useState(initialData);
  const [isPending, startTransition] = useTransition();

  const isUp = useMemo(() => {
    if (data.length < 2) return true;
    return data[data.length - 1].price >= data[0].price;
  }, [data]);

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

  return (
    <div>
      <div className="mb-2 flex justify-end gap-1">
        {RANGES.map((range) => (
          <button
            key={range.period}
            type="button"
            onClick={() => handleRangeChange(range.period)}
            className={`rounded-md px-2 py-1 text-xs font-medium transition-colors ${
              period === range.period
                ? "bg-primary text-primary-foreground"
                : "text-muted-foreground hover:bg-muted"
            }`}
          >
            {range.label}
          </button>
        ))}
      </div>

      {data.length === 0 ? (
        <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
          No price data available
        </div>
      ) : (
        <div className={`h-64 w-full ${isPending ? "opacity-60" : ""}`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <XAxis dataKey="date" hide />
              <YAxis domain={["auto", "auto"]} hide />
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: "var(--radius-md)",
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
