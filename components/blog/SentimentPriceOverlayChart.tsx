"use client";

import {
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";

interface SentimentPriceOverlayChartProps {
  data: SentimentPricePoint[];
}

export function SentimentPriceOverlayChart({ data }: SentimentPriceOverlayChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        Not enough sentiment history yet for this chart.
      </div>
    );
  }

  // Axes are hidden, so summarize the two series in words for screen readers.
  const withSentiment = data.filter((d) => d.averageSentiment != null);
  const withPrice = data.filter((d) => d.price != null);
  const summaryParts: string[] = [];
  if (withSentiment.length >= 2) {
    const first = withSentiment[0];
    const last = withSentiment[withSentiment.length - 1];
    summaryParts.push(
      `Average news sentiment (scale −1 to 1) went from ${first.averageSentiment!.toFixed(2)} on ${first.date} to ${last.averageSentiment!.toFixed(2)} on ${last.date}.`
    );
  }
  if (withPrice.length >= 2) {
    const first = withPrice[0];
    const last = withPrice[withPrice.length - 1];
    summaryParts.push(
      `Price went from $${first.price!.toFixed(2)} on ${first.date} to $${last.price!.toFixed(2)} on ${last.date}.`
    );
  }

  return (
    <div className="w-full">
      {summaryParts.length > 0 && <p className="sr-only">{summaryParts.join(" ")}</p>}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" hide />
            <YAxis yAxisId="sentiment" domain={[-1, 1]} hide />
            <YAxis yAxisId="price" orientation="right" domain={["auto", "auto"]} hide />
            <Tooltip
              contentStyle={{
                background: "var(--color-popover)",
                border: "1px solid var(--color-border)",
                borderRadius: 0,
                color: "var(--color-popover-foreground)",
                fontSize: 12,
              }}
              formatter={(value, name) =>
                name === "price" && typeof value === "number"
                  ? [`$${value.toFixed(2)}`, "Price"]
                  : [value, name]
              }
            />
            <Line
              yAxisId="sentiment"
              type="monotone"
              dataKey="averageSentiment"
              stroke="var(--color-chart-2)"
              strokeWidth={2}
              dot={{ r: 3, fill: "var(--color-chart-2)", strokeWidth: 0 }}
              connectNulls
              isAnimationActive={false}
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="price"
              stroke="var(--color-chart-1)"
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex justify-center gap-4 text-xs text-muted-foreground">
        <span>
          <span className="inline-block h-2 w-2 rounded-none bg-[var(--color-chart-1)]" /> Price
        </span>
        <span>
          <span className="inline-block h-2 w-2 rounded-none bg-[var(--color-chart-2)]" /> Sentiment
        </span>
      </div>
    </div>
  );
}
