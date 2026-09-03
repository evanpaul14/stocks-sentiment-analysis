import { fetchTrendingSymbols } from "@/lib/integrations/stocktwits/trending";
import { fetchRedditTrending, type RedditTrendingItem } from "@/lib/integrations/apewisdom";
import { fetchVolumeTrending } from "@/lib/integrations/alpaca";
import { getPriceSnapshots } from "@/lib/integrations/yahoo/quote";
import { TtlCache } from "@/lib/cache/memory";

export type TrendingSource = "stocktwits" | "reddit" | "volume";

export interface TrendingItem {
  symbol: string;
  companyName: string;
  price: number | null;
  changePercent: number | null;
  meta:
    | { source: "stocktwits"; watchlistCount: number }
    | { source: "reddit"; mentions: number; mentionsChangePercent: number; tag: RedditTrendingItem["tag"] }
    | { source: "volume"; volume: number; tradeCount: number };
}

async function enrichWithPrices<T extends { symbol: string; fallbackName?: string }>(
  items: T[],
  toItem: (item: T, price: number | null, changePercent: number | null, companyName: string) => TrendingItem
): Promise<TrendingItem[]> {
  if (items.length === 0) return [];
  const snapshots = await getPriceSnapshots(items.map((i) => i.symbol));
  const bySymbol = new Map(snapshots.map((s) => [s.symbol, s]));

  return items.map((item) => {
    const snapshot = bySymbol.get(item.symbol);
    const companyName = snapshot?.companyName ?? item.fallbackName ?? item.symbol;
    return toItem(item, snapshot?.price ?? null, snapshot?.changePercent ?? null, companyName);
  });
}

async function getStockTwitsTrending(): Promise<TrendingItem[]> {
  const symbols = await fetchTrendingSymbols(10).catch(() => []);
  return enrichWithPrices(
    symbols.map((s) => ({ symbol: s.symbol, fallbackName: s.title })),
    (item, price, changePercent, companyName) => ({
      symbol: item.symbol,
      companyName,
      price,
      changePercent,
      meta: {
        source: "stocktwits",
        watchlistCount: symbols.find((s) => s.symbol === item.symbol)?.watchlist_count ?? 0,
      },
    })
  );
}

async function getRedditTrendingEnriched(): Promise<TrendingItem[]> {
  const items = await fetchRedditTrending(10);
  return enrichWithPrices(
    items.map((i) => ({ symbol: i.ticker, fallbackName: i.name })),
    (item, price, changePercent, companyName) => {
      const source = items.find((i) => i.ticker === item.symbol)!;
      return {
        symbol: item.symbol,
        companyName,
        price,
        changePercent,
        meta: {
          source: "reddit",
          mentions: source.mentions,
          mentionsChangePercent: source.mentionsChangePercent,
          tag: source.tag,
        },
      };
    }
  );
}

async function getVolumeTrendingEnriched(): Promise<TrendingItem[]> {
  const items = await fetchVolumeTrending();
  return enrichWithPrices(
    items.map((i) => ({ symbol: i.symbol })),
    (item, price, changePercent, companyName) => {
      const source = items.find((i) => i.symbol === item.symbol)!;
      return {
        symbol: item.symbol,
        companyName,
        price,
        changePercent,
        meta: { source: "volume", volume: source.volume, tradeCount: source.tradeCount },
      };
    }
  );
}

const FETCHERS: Record<TrendingSource, () => Promise<TrendingItem[]>> = {
  stocktwits: getStockTwitsTrending,
  reddit: getRedditTrendingEnriched,
  volume: getVolumeTrendingEnriched,
};

const TRENDING_CACHE_TTL_MS = 90_000;
const trendingCache = new TtlCache<TrendingItem[]>(TRENDING_CACHE_TTL_MS);

/** Single-source trending data. Degrades to [] on any upstream failure. */
export async function getTrendingSourceData(source: TrendingSource): Promise<TrendingItem[]> {
  return trendingCache.getOrCompute(source, async () => {
    try {
      return await FETCHERS[source]();
    } catch (error) {
      console.error(`[trending] source "${source}" failed`, error);
      return [];
    }
  });
}

/** All three trending sources, fetched in parallel. */
export async function getAllTrendingSourceData(): Promise<Record<TrendingSource, TrendingItem[]>> {
  const [stocktwits, reddit, volume] = await Promise.all([
    getTrendingSourceData("stocktwits"),
    getTrendingSourceData("reddit"),
    getTrendingSourceData("volume"),
  ]);
  return { stocktwits, reddit, volume };
}
