import { stocktwitsFetch } from "./fetchClient";

const TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json";
const ALLOWED_INSTRUMENT_CLASSES = new Set(["stock", "exchangetradedcommodity"]);

export interface TrendingSymbolPayload {
  symbol: string;
  title: string;
  exchange: string | null;
  instrument_class: string;
  watchlist_count: number;
  trends?: { summary?: string };
}

interface TrendingResponse {
  symbols: Array<{
    symbol: string;
    title: string;
    exchange: string | null;
    instrument_class: string;
    watchlist_count: number;
    trends?: { summary?: string };
  }>;
}

/** Trending symbols, filtered to stocks/ETFs (excludes crypto). */
export async function fetchTrendingSymbols(
  limit = 60
): Promise<TrendingSymbolPayload[]> {
  const data = (await stocktwitsFetch(TRENDING_URL)) as TrendingResponse;
  return data.symbols
    .filter(
      (s) =>
        s.exchange !== "CRYPTO" &&
        ALLOWED_INSTRUMENT_CLASSES.has(s.instrument_class?.toLowerCase())
    )
    .slice(0, limit);
}
