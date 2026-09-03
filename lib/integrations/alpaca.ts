const ALPACA_URL =
  "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=10";

export interface VolumeTrendingItem {
  symbol: string;
  volume: number;
  tradeCount: number;
}

interface AlpacaMostActivesResponse {
  most_actives?: Array<{ symbol: string; volume: number; trade_count: number }>;
}

function isEnabled(): boolean {
  return Boolean(
    (process.env.ALPACA_API_KEY_ID || process.env.ALPACA_API_KEY) &&
      (process.env.ALPACA_API_SECRET_KEY || process.env.ALPACA_SECRET_KEY)
  );
}

/**
 * Most-active-by-volume symbols from Alpaca. Note: this endpoint doesn't
 * return price/name — callers should enrich with a Yahoo quote batch.
 * Degrades to [] on failure or missing credentials.
 */
export async function fetchVolumeTrending(): Promise<VolumeTrendingItem[]> {
  if (!isEnabled()) return [];

  try {
    const response = await fetch(ALPACA_URL, {
      headers: {
        "Apca-Api-Key-Id": (process.env.ALPACA_API_KEY_ID || process.env.ALPACA_API_KEY)!,
        "Apca-Api-Secret-Key": (process.env.ALPACA_API_SECRET_KEY || process.env.ALPACA_SECRET_KEY)!,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return [];

    const data: AlpacaMostActivesResponse = await response.json();
    return (data.most_actives ?? []).map((item) => ({
      symbol: item.symbol,
      volume: item.volume,
      tradeCount: item.trade_count,
    }));
  } catch (error) {
    console.error("[alpaca] most-actives fetch failed", error);
    return [];
  }
}
