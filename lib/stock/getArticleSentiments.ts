import type { NewsArticle } from "@/lib/integrations/news/googleNews";
import { classifySentimentBatch } from "@/lib/integrations/sentiment/classify";
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
  const cached = await Promise.all(
    articles.map((article) =>
      sentimentHistory.findExisting(ticker, article.link || null)
    )
  );

  const pending = articles.filter((_, i) => !cached[i]);
  const classified =
    pending.length > 0
      ? await classifySentimentBatch(
          companyName,
          pending.map((article) => ({
            title: article.title,
            description: article.description ?? "",
          }))
        )
      : [];

  let nextPending = 0;
  return Promise.all(
    articles.map(async (article, i): Promise<ArticleSentimentResult> => {
      const existing = cached[i];
      if (existing) return { article, sentiment: existing.sentiment };

      const result = classified[nextPending++];
      if (result.status === "rejected") {
        console.error(`[getArticleSentiments] classification failed for ${ticker}`, result.reason);
        return { article, sentiment: "error" };
      }

      try {
        const sentiment = result.value;
        const row = await sentimentHistory.insert({
          ticker,
          articleTitle: article.title,
          articleLink: article.link || null,
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
