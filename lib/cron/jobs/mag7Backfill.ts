import { getNewsArticles } from "@/lib/integrations/news/googleNews";
import { classifySentiment } from "@/lib/integrations/sentiment/classify";
import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";
import { SEO_SENTIMENT_COMPANIES } from "@/lib/utils/tickers";

/**
 * Weekly backfill: classify+persist fresh sentiment for every SEO sentiment company/index,
 * so pages get sentiment data even without organic /stock/<ticker> traffic (the mechanism
 * individual stock pages otherwise rely on). Originally MAG7-only, hence the file/job name.
 */
export async function runMag7SentimentJob() {
  for (const { ticker, companyName } of SEO_SENTIMENT_COMPANIES) {
    try {
      // Search by company name, not raw ticker — index tickers like "^GSPC" make a poor
      // Google News query, while "S&P 500" reliably surfaces relevant coverage.
      const articles = await getNewsArticles(companyName, 10);
      for (const article of articles) {
        const existing = await sentimentHistory.findExisting(ticker, article.link);
        if (existing) continue;

        const sentiment = await classifySentiment(companyName, article.title, article.description);
        await sentimentHistory.insert({
          ticker,
          articleTitle: article.title,
          articleLink: article.link,
          articleSource: article.source,
          articlePublishedAt: article.publishedAt,
          sentiment,
        });
      }
    } catch (error) {
      console.error(`[mag7-backfill] failed for ${ticker}`, error);
    }
  }
}
