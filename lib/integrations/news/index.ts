import { TtlCache } from "@/lib/cache/memory";
import { fetchGoogleNews } from "./googleNews";
import { fetchYahooNews } from "./yahooNews";
import { fetchAlpacaNews } from "./alpacaNews";
import type { NewsArticle } from "./types";

export type { NewsArticle } from "./types";
export { getMarketNewsDigest } from "./googleNews";

const ARTICLES_CACHE_TTL_MS =
  Number(process.env.NEWS_ARTICLES_CACHE_TTL_SECONDS ?? 600) * 1000;
const articlesCache = new TtlCache<NewsArticle[]>(ARTICLES_CACHE_TTL_MS);

/** A source returning fewer than this is treated as too thin; keep trying, keep the best. */
const MIN_USEFUL_ARTICLES = 3;

/** Tried in order; the first source with a useful number of articles wins (otherwise the fullest result is used). */
const SOURCES: Array<{
  name: string;
  fetch: (query: string, limit: number) => Promise<NewsArticle[]>;
}> = [
  { name: "google", fetch: fetchGoogleNews },
  { name: "yahoo", fetch: fetchYahooNews },
  { name: "alpaca", fetch: fetchAlpacaNews },
];

/**
 * Per-ticker news, used by /api/search (sentiment excluded, kept fast) and
 * the stock page's sentiment stream. Falls back Google News -> Yahoo -> Alpaca
 * when a source errors (e.g. Google's 429s) or comes back empty. Cached per
 * query+limit so reloading the same stock page doesn't re-fetch the feed (or
 * re-trigger the sentiment stream's per-article classification loop) within
 * the TTL; empty results are not cached so an outage doesn't stick.
 */
export async function getNewsArticles(
  symbol: string,
  limit = 10
): Promise<NewsArticle[]> {
  const key = `${symbol.toUpperCase()}:${limit}`;
  const articles = await articlesCache.getOrCompute(key, async () => {
    let best: NewsArticle[] = [];
    for (const source of SOURCES) {
      try {
        const result = await source.fetch(symbol, limit);
        if (result.length >= Math.min(MIN_USEFUL_ARTICLES, limit)) return result;
        if (result.length > best.length) best = result;
        console.warn(`[news] ${source.name} returned ${result.length} articles for ${symbol}`);
      } catch (error) {
        console.error(`[news] ${source.name} failed for ${symbol}`, error);
      }
    }
    return best;
  });
  if (articles.length === 0) articlesCache.delete(key);
  return articles;
}
