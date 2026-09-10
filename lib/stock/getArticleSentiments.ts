import type { NewsArticle } from "@/lib/integrations/news/googleNews";
import { classifySentiment } from "@/lib/integrations/sentiment/classify";
import type { SentimentLabel } from "@/lib/integrations/sentiment/classify";
import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";

export interface ArticleSentimentResult {
  article: NewsArticle;
  sentiment: SentimentLabel | "error";
}

/**
 * Classifies every article server-side so the initial HTML response (what
 * crawlers and AI bots see) already contains sentiment counts, instead of
 * the client discovering them one at a time after hydration. Mirrors the
 * cache-then-classify logic in `app/api/sentiment/route.ts`.
 */
export async function getArticleSentiments(
  ticker: string,
  companyName: string,
  articles: NewsArticle[]
): Promise<ArticleSentimentResult[]> {
  return Promise.all(
    articles.map(async (article): Promise<ArticleSentimentResult> => {
      const link = article.link || null;
      const existing = await sentimentHistory.findExisting(ticker, link);
      if (existing) return { article, sentiment: existing.sentiment };

      try {
        const sentiment = await classifySentiment(
          companyName,
          article.title,
          article.description ?? ""
        );
        const row = await sentimentHistory.insert({
          ticker,
          articleTitle: article.title,
          articleLink: link,
          articleSource: article.source ?? null,
          articlePublishedAt: article.publishedAt ?? null,
          sentiment,
        });
        return { article, sentiment: row?.sentiment ?? sentiment };
      } catch (error) {
        console.error(`[getArticleSentiments] classification failed for ${ticker}`, error);
        return { article, sentiment: "error" };
      }
    })
  );
}

export type SentimentVerdictLabel = "Bullish" | "Bearish" | "Neutral";

export interface SentimentVerdict {
  label: SentimentVerdictLabel;
  positive: number;
  negative: number;
  neutral: number;
  total: number;
}

/**
 * Rolls per-article sentiment into the single above-the-fold verdict every
 * ranking competitor leads with (see seo_report.md #10) — positive/negative
 * imbalance beyond +/-20% of analyzed articles tips the label; otherwise
 * "Neutral". Returns null when nothing was successfully classified yet.
 */
export function computeSentimentVerdict(
  results: ArticleSentimentResult[]
): SentimentVerdict | null {
  const counts = { positive: 0, negative: 0, neutral: 0 };
  for (const { sentiment } of results) {
    if (sentiment === "error") continue;
    counts[sentiment]++;
  }

  const total = counts.positive + counts.negative + counts.neutral;
  if (total === 0) return null;

  const score = (counts.positive - counts.negative) / total;
  const label: SentimentVerdictLabel =
    score > 0.2 ? "Bullish" : score < -0.2 ? "Bearish" : "Neutral";

  return { label, ...counts, total };
}
