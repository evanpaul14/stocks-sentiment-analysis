import { cache } from "react";
import { resolveSymbol } from "@/lib/integrations/yahoo/search";
import { getStockInfo, type StockInfo } from "@/lib/integrations/yahoo/quote";
import { getHistoricalPrices, type PricePoint } from "@/lib/integrations/yahoo/historical";
import { getNewsArticles, type NewsArticle } from "@/lib/integrations/news";

export interface StockPageData {
  symbol: string;
  stockInfo: StockInfo;
  historicalData: PricePoint[];
  articles: NewsArticle[];
}

/**
 * Resolves a query (ticker or company name) and loads everything the stock
 * detail page needs for its initial server render, in parallel.
 *
 * Wrapped in React's `cache` so generateMetadata and the page component
 * share one lookup per request instead of each hitting Yahoo.
 */
export const getStockPageData = cache(async (rawQuery: string): Promise<StockPageData> => {
  const symbol = await resolveSymbol(rawQuery);

  const [stockInfo, historicalData, articles] = await Promise.all([
    getStockInfo(symbol),
    getHistoricalPrices(symbol, "1d"),
    getNewsArticles(symbol, 10),
  ]);

  return { symbol, stockInfo, historicalData, articles };
});
