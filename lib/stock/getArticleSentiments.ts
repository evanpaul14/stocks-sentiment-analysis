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
