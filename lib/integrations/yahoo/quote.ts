import { yahooFinance } from "./client";

export interface StockInfo {
  symbol: string;
  companyName: string;
  currentPrice: number | null;
  previousClose: number | null;
  regularMarketChangePercent: number | null;
  postMarketPrice: number | null;
  postMarketChange: number | null;
  postMarketChangePercent: number | null;
  marketCap: number | null;
  peRatio: number | null;
  dividendYield: number | null;
  averageDailyVolume10Day: number | null;
  dayHigh: number | null;
  dayLow: number | null;
  open: number | null;
  volume: number | null;
  fiftyTwoWeekHigh: number | null;
  fiftyTwoWeekLow: number | null;
  ceoName: string | null;
  fullTimeEmployees: number | null;
  city: string | null;
  state: string | null;
  country: string | null;
  industry: string | null;
  sector: string | null;
  website: string | null;
  businessSummary: string | null;
  yearFounded: number | null;
}

/** Full quote payload for the stock detail page — mirrors the old app's get_stock_info(). */
export async function getStockInfo(symbol: string): Promise<StockInfo> {
  const result = await yahooFinance.quoteSummary(symbol, {
    modules: ["price", "summaryDetail", "assetProfile"],
  });

  const price = result.price;
  const summary = result.summaryDetail;
  const profile = result.assetProfile;

  const ceo = profile?.companyOfficers?.find((officer) =>
    officer.title?.toLowerCase().includes("ceo")
  );

  const yearFounded = extractYearFounded(profile?.longBusinessSummary);

  const currentPrice = price?.regularMarketPrice ?? null;
  const previousClose =
    summary?.previousClose ?? price?.regularMarketPreviousClose ?? null;
  const postMarketPrice = price?.postMarketPrice ?? null;
  const postMarketChange = price?.postMarketChange ?? null;

  return {
    symbol,
    companyName: price?.longName ?? price?.shortName ?? symbol,
    currentPrice,
    previousClose,
    // Derived directly from prices rather than trusted from Yahoo's
    // quoteSummary "price" module, whose *ChangePercent fields come back
    // as raw fractions (e.g. -0.0063) instead of percent (-0.63) here.
    regularMarketChangePercent: percentChange(currentPrice, previousClose),
    postMarketPrice,
    postMarketChange,
    postMarketChangePercent: percentChange(postMarketPrice, currentPrice),
    marketCap: summary?.marketCap ?? price?.marketCap ?? null,
    peRatio: summary?.forwardPE ?? summary?.trailingPE ?? null,
    dividendYield: summary?.dividendYield ?? null,
    averageDailyVolume10Day:
      summary?.averageDailyVolume10Day ?? price?.averageDailyVolume10Day ?? null,
    dayHigh: summary?.dayHigh ?? price?.regularMarketDayHigh ?? null,
    dayLow: summary?.dayLow ?? price?.regularMarketDayLow ?? null,
    open: summary?.open ?? price?.regularMarketOpen ?? null,
    volume: summary?.volume ?? price?.regularMarketVolume ?? null,
    fiftyTwoWeekHigh: summary?.fiftyTwoWeekHigh ?? null,
    fiftyTwoWeekLow: summary?.fiftyTwoWeekLow ?? null,
    ceoName: ceo?.name ?? null,
    fullTimeEmployees: profile?.fullTimeEmployees ?? null,
    city: profile?.city ?? null,
    state: profile?.state ?? null,
    country: profile?.country ?? null,
    industry: profile?.industry ?? null,
    sector: profile?.sector ?? null,
    website: profile?.website ?? null,
    businessSummary: profile?.longBusinessSummary ?? null,
    yearFounded,
  };
}

function percentChange(current: number | null, base: number | null): number | null {
  if (current == null || base == null || base === 0) return null;
  return ((current - base) / base) * 100;
}

function extractYearFounded(businessSummary?: string): number | null {
  if (!businessSummary) return null;
  const match = businessSummary.match(/\bin (19|20)\d{2}\b/);
  if (!match) return null;
  const year = Number(match[0].slice(3));
  return Number.isFinite(year) ? year : null;
}

export interface PriceSnapshot {
  symbol: string;
  companyName: string | null;
  price: number | null;
  previousClose: number | null;
  changePercent: number | null;
}

/** Lightweight price+change snapshot for live polling (watchlist, search results). */
export async function getPriceSnapshot(symbol: string): Promise<PriceSnapshot> {
  const quote = await yahooFinance.quote(symbol);
  return {
    symbol,
    companyName: quote.longName ?? quote.shortName ?? null,
    price: quote.regularMarketPrice ?? null,
    previousClose: quote.regularMarketPreviousClose ?? null,
    changePercent: quote.regularMarketChangePercent ?? null,
  };
}

/** Batched snapshot for watchlist/trending price refresh. */
export async function getPriceSnapshots(
  symbols: string[]
): Promise<PriceSnapshot[]> {
  if (symbols.length === 0) return [];
  const quotes = await yahooFinance.quote(symbols);
  return quotes.map((quote) => ({
    symbol: quote.symbol,
    companyName: quote.longName ?? quote.shortName ?? null,
    price: quote.regularMarketPrice ?? null,
    previousClose: quote.regularMarketPreviousClose ?? null,
    changePercent: quote.regularMarketChangePercent ?? null,
  }));
}
