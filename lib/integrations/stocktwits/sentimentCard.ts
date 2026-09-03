import { collectStockTwitsMessages } from "./stream";
import { calculateStockTwitsSentiment, type StockTwitsSentimentResult } from "./sentiment";
import { buildStockTwitsMessageFeed, type FeedMessage } from "./media";
import { TtlCache } from "@/lib/cache/memory";

export interface StockTwitsSentimentCard {
  sentiment: StockTwitsSentimentResult;
  feed: FeedMessage[];
}

const CACHE_TTL_MS =
  Number(process.env.STOCKTWITS_SYMBOL_SENTIMENT_CACHE_TTL_SECONDS ?? 180) * 1000;
const cardCache = new TtlCache<StockTwitsSentimentCard>(CACHE_TTL_MS);

/** Bullish/bearish ratios + message feed for a ticker's StockTwits sentiment card. */
export async function getStockTwitsSentimentCard(
  symbol: string
): Promise<StockTwitsSentimentCard> {
  return cardCache.getOrCompute(symbol.toUpperCase(), async () => {
    const messages = await collectStockTwitsMessages(symbol);
    return {
      sentiment: calculateStockTwitsSentiment(messages),
      feed: buildStockTwitsMessageFeed(messages, 40),
    };
  });
}
