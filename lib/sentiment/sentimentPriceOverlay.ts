import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";
import { getHistoricalPrices } from "@/lib/integrations/yahoo/historical";

const SENTIMENT_SCORE: Record<string, number> = { positive: 1, neutral: 0, negative: -1 };

export interface SentimentPricePoint {
  date: string;
  averageSentiment: number | null;
  articleCount: number;
  price: number | null;
}

function formatDateLabel(dateKey: string): string {
  return new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
  }).format(new Date(`${dateKey}T00:00:00Z`));
}

/**
 * Plain-text description of the sentiment-vs-price overlay chart, which is
 * otherwise canvas/SVG-only and has no text an AI crawler (or a screen
 * reader) can extract. Meant to be rendered alongside the chart, not
 * replace it.
 */
export function summarizeOverlay(overlay: SentimentPricePoint[], displayTicker: string): string {
  const withPrice = overlay.filter((p) => p.price != null);
  const withSentiment = overlay.filter((p) => p.averageSentiment != null);

  if (withPrice.length < 2 && withSentiment.length < 2) {
    return `Not enough sentiment or price history yet to summarize a trend for ${displayTicker}.`;
  }

  const parts: string[] = [];

  if (withPrice.length >= 2) {
    const first = withPrice[0];
    const last = withPrice[withPrice.length - 1];
    const changePercent = (((last.price ?? 0) - (first.price ?? 0)) / (first.price ?? 1)) * 100;
    parts.push(
      `From ${formatDateLabel(first.date)} to ${formatDateLabel(last.date)}, ${displayTicker} ${
        changePercent >= 0 ? "rose" : "fell"
      } ${Math.abs(changePercent).toFixed(2)}% (from $${(first.price ?? 0).toFixed(2)} to $${(last.price ?? 0).toFixed(2)}).`
    );
  }

  if (withSentiment.length >= 2) {
    const first = withSentiment[0];
    const last = withSentiment[withSentiment.length - 1];
    const label = (score: number) => (score > 0.2 ? "positive" : score < -0.2 ? "negative" : "neutral");
    parts.push(
      `News sentiment over the same window moved from ${label(first.averageSentiment ?? 0)} (${(first.averageSentiment ?? 0).toFixed(2)}) to ${label(last.averageSentiment ?? 0)} (${(last.averageSentiment ?? 0).toFixed(2)}) on a -1 to +1 scale.`
    );
  }

  return parts.join(" ");
}

/** Last 90 days of daily-averaged sentiment, joined against 3-month price history. */
export async function getSentimentPriceOverlay(ticker: string): Promise<SentimentPricePoint[]> {
  const sinceIso = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString();

  const [history, prices] = await Promise.all([
    sentimentHistory.listSince(ticker.toUpperCase(), sinceIso),
    getHistoricalPrices(ticker.toUpperCase(), "3mo"),
  ]);

  const scoresByDate = new Map<string, number[]>();
  for (const row of history) {
    const dateKey = row.analyzedAt.slice(0, 10);
    const score = SENTIMENT_SCORE[row.sentiment];
    const list = scoresByDate.get(dateKey) ?? [];
    list.push(score);
    scoresByDate.set(dateKey, list);
  }

  const priceByDate = new Map<string, number>();
  for (const point of prices) {
    priceByDate.set(point.date.slice(0, 10), point.price);
  }

  const allDates = new Set([...scoresByDate.keys(), ...priceByDate.keys()]);

  return [...allDates]
    .sort()
    .map((date) => {
      const scores = scoresByDate.get(date);
      return {
        date,
        averageSentiment: scores
          ? Number((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3))
          : null,
        articleCount: scores?.length ?? 0,
        price: priceByDate.get(date) ?? null,
      };
    });
}
