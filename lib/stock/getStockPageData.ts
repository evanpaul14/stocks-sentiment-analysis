import { resolveSymbol } from "@/lib/integrations/yahoo/search";
import { getStockInfo, type StockInfo } from "@/lib/integrations/yahoo/quote";
import { getHistoricalPrices, type PricePoint } from "@/lib/integrations/yahoo/historical";
import { getNewsArticles, type NewsArticle } from "@/lib/integrations/news/googleNews";

export interface StockPageData {
  symbol: string;
  stockInfo: StockInfo;
  historicalData: PricePoint[];
  articles: NewsArticle[];
}

/**
 * Resolves a query (ticker or company name) and loads everything the stock
 * detail page needs for its initial server render, in parallel.
 */
export async function getStockPageData(rawQuery: string): Promise<StockPageData> {
  const symbol = await resolveSymbol(rawQuery);

  const [stockInfo, historicalData, articles] = await Promise.all([
    getStockInfo(symbol),
    getHistoricalPrices(symbol, "1d"),
    getNewsArticles(symbol, 10),
  ]);

  return { symbol, stockInfo, historicalData, articles };
}
