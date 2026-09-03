import { yahooFinance } from "./client";

export type ChartInterval =
  | "1m"
  | "5m"
  | "15m"
  | "30m"
  | "60m"
  | "90m"
  | "1h"
  | "1d"
  | "5d"
  | "1wk"
  | "1mo"
  | "3mo";

export interface PricePoint {
  date: string;
  price: number;
}

const PERIOD_TO_DEFAULT_INTERVAL: Record<string, ChartInterval> = {
  "1d": "5m",
  "5d": "15m",
  "1mo": "90m",
};

const PERIOD_DAYS: Record<string, number> = {
  "1mo": 30,
  "3mo": 90,
  "6mo": 180,
  "1y": 365,
  "2y": 730,
  "5y": 1825,
  max: 20 * 365,
};

function daysAgo(days: number): Date {
  return new Date(Date.now() - days * 24 * 60 * 60 * 1000);
}

function toDateKey(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/**
 * Ports the old app's yfinance period/interval quirks:
 * - "7d"/"1w" remap to "5d"
 * - interval defaults: 1d->5m, 5d->15m, 1mo->90m
 * - period=1d fetches 5d@5m then filters down to only the most recent
 *   calendar date (Yahoo doesn't reliably return true "1 day" intraday data)
 * - period=5d fetches 5d@15m
 * - anything else passes through as a plain period1=now-N-days lookback
 */
export async function getHistoricalPrices(
  symbol: string,
  requestedPeriod: string,
  requestedInterval?: ChartInterval
): Promise<PricePoint[]> {
  const period = requestedPeriod === "7d" || requestedPeriod === "1w"
    ? "5d"
    : requestedPeriod;

  if (period === "1d") {
    const result = await yahooFinance.chart(symbol, {
      period1: daysAgo(5),
      interval: "5m",
    });
    const rows = result.quotes.filter((q) => q.close != null && q.date);
    if (rows.length === 0) return [];
    const lastDateKey = toDateKey(new Date(rows[rows.length - 1].date));
    return rows
      .filter((q) => toDateKey(new Date(q.date)) === lastDateKey)
      .map((q) => ({ date: formatTimestamp(q.date), price: q.close as number }));
  }

  if (period === "5d") {
    const result = await yahooFinance.chart(symbol, {
      period1: daysAgo(5),
      interval: "15m",
    });
    return toPricePoints(result.quotes);
  }

  const interval =
    requestedInterval ?? PERIOD_TO_DEFAULT_INTERVAL[period] ?? "1d";
  const days = PERIOD_DAYS[period] ?? 30;
  const result = await yahooFinance.chart(symbol, {
    period1: daysAgo(days),
    interval,
  });
  return toPricePoints(result.quotes);
}

function toPricePoints(
  quotes: Array<{ date: Date; close: number | null }>
): PricePoint[] {
  return quotes
    .filter((q) => q.close != null)
    .map((q) => ({ date: formatTimestamp(q.date), price: q.close as number }));
}

function formatTimestamp(date: Date): string {
  return date.toISOString().replace("T", " ").slice(0, 19);
}
