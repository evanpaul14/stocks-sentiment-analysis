import { fetchTrendingSymbols } from "./trending";
import { TtlCache } from "@/lib/cache/memory";

const SUMMARY_CACHE_TTL_MS =
  Number(process.env.STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS ?? 600) * 1000;
const summaryCache = new TtlCache<string | null>(SUMMARY_CACHE_TTL_MS);

/** Cached one-line AI/StockTwits-derived blurb for hover popovers in trending lists. */
export async function getStockTwitsSummary(symbol: string): Promise<string | null> {
  return summaryCache.getOrCompute(symbol.toUpperCase(), async () => {
    const trending = await fetchTrendingSymbols(200);
    const match = trending.find((s) => s.symbol === symbol.toUpperCase());
    return match?.trends?.summary ?? null;
  });
}
