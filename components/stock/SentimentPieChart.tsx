"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

export type SentimentLabel = "positive" | "negative" | "neutral";

export const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
  positive: "var(--color-chart-1)",
  negative: "var(--color-destructive)",
  neutral: "var(--color-muted-foreground)",
};

export function SentimentPieChart({
  chartData,
}: {
  chartData: { name: SentimentLabel; value: number }[];
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie
          data={chartData}
          dataKey="value"
          innerRadius={28}
          outerRadius={44}
          isAnimationActive={false}
        >
          {chartData.map((entry) => (
            <Cell key={entry.name} fill={SENTIMENT_COLORS[entry.name]} />
          ))}
        </Pie>
      </PieChart>
    </ResponsiveContainer>
  );
}
