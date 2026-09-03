import { getNewsArticles } from "@/lib/integrations/news/googleNews";
import { classifySentiment } from "@/lib/integrations/sentiment/classify";
import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";

export const MAG7_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"];

/** Weekly backfill: classify+persist fresh sentiment for each Magnificent 7 ticker. */
export async function runMag7SentimentJob() {
  for (const ticker of MAG7_TICKERS) {
    try {
      const articles = await getNewsArticles(ticker, 10);
      for (const article of articles) {
        const existing = await sentimentHistory.findExisting(ticker, article.link);
        if (existing) continue;

        const sentiment = await classifySentiment(ticker, article.title, article.description);
        sentimentHistory.insert({
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
