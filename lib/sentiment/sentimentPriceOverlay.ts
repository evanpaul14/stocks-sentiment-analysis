import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";
import { getHistoricalPrices } from "@/lib/integrations/yahoo/historical";

const SENTIMENT_SCORE: Record<string, number> = { positive: 1, neutral: 0, negative: -1 };

export interface SentimentPricePoint {
  date: string;
  averageSentiment: number | null;
  articleCount: number;
  price: number | null;
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
