import { yahooFinance } from "./client";

export const MARKET_SUMMARY_INDEXES = [
  { symbol: "^GSPC", name: "S&P 500" },
  { symbol: "^IXIC", name: "Nasdaq Composite" },
  { symbol: "^DJI", name: "Dow Jones Industrial Average" },
];

export interface IndexSnapshot {
  symbol: string;
  name: string;
  price: number | null;
  changePercent: number | null;
  weekChangePercent: number | null;
}

async function getWeekChangePercent(symbol: string): Promise<number | null> {
  try {
    const result = await yahooFinance.chart(symbol, {
      period1: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000),
      interval: "1d",
    });
    const closes = result.quotes
      .filter((q) => q.close != null)
      .map((q) => q.close as number);
    if (closes.length < 2) return null;
    const first = closes[0];
    const last = closes[closes.length - 1];
    if (first === 0) return null;
    return ((last - first) / first) * 100;
  } catch {
    return null;
  }
}

/** Index snapshots (price + day/week change) for the daily market summary. */
export async function getMarketIndexSnapshots(): Promise<IndexSnapshot[]> {
  const symbols = MARKET_SUMMARY_INDEXES.map((i) => i.symbol);
  const quotes = await yahooFinance.quote(symbols);
  const bySymbol = new Map(quotes.map((q) => [q.symbol, q]));

  return Promise.all(
    MARKET_SUMMARY_INDEXES.map(async ({ symbol, name }) => {
      const quote = bySymbol.get(symbol);
      return {
        symbol,
        name,
        price: quote?.regularMarketPrice ?? null,
        changePercent: quote?.regularMarketChangePercent ?? null,
        weekChangePercent: await getWeekChangePercent(symbol),
      };
    })
  );
}
