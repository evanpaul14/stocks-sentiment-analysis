import type { SentimentVerdict } from "@/lib/stock/getArticleSentiments";

const VERDICT_COLORS: Record<SentimentVerdict["label"], string> = {
  Bullish: "var(--color-chart-1)",
  Bearish: "var(--color-destructive)",
  Neutral: "var(--color-muted-foreground)",
};

export function SentimentVerdictBadge({ verdict }: { verdict: SentimentVerdict }) {
  return (
    <span
      className="inline-flex w-fit items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-medium"
      style={{ color: VERDICT_COLORS[verdict.label], borderColor: VERDICT_COLORS[verdict.label] }}
    >
      <span
        className="size-2 rounded-full"
        style={{ backgroundColor: VERDICT_COLORS[verdict.label] }}
      />
      {verdict.label} News Sentiment
    </span>
  );
}
