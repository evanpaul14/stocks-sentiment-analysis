import { yahooFinance } from "./client";

/**
 * Mirrors the old app's `normalize_search_query`: only uppercase + truncate
 * at the first "." when the input has no spaces (i.e. looks like a ticker,
 * e.g. "brk.b" -> "BRK"). Free-text company names pass through unmodified.
 */
export function normalizeSearchQuery(rawQuery: string): string {
  const trimmed = rawQuery.trim();
  if (trimmed.includes(" ")) return trimmed;
  const dotIndex = trimmed.indexOf(".");
  const truncated = dotIndex === -1 ? trimmed : trimmed.slice(0, dotIndex);
  return truncated.toUpperCase();
}

export class SymbolNotFoundError extends Error {
  constructor(query: string) {
    super(`Company not found for query: ${query}`);
    this.name = "SymbolNotFoundError";
  }
}

/** Yahoo exchange codes for US-listed securities (NYSE, Nasdaq, NYSE American/Arca, Cboe). */
const US_EXCHANGES = new Set([
  "NMS", "NGM", "NCM", "NIM", "NYQ", "ASE", "PCX", "BTS", "BATS", "NYS",
]);

/**
 * Resolves a free-text company/ticker query to a Yahoo Finance symbol.
 * Only equities and ETFs are considered; US-listed results are preferred,
 * falling back to the first non-US equity/ETF if no US match exists.
 */
export async function resolveSymbol(rawQuery: string): Promise<string> {
  const query = normalizeSearchQuery(rawQuery);
  const results = await yahooFinance.search(query);
  const candidates = results.quotes.filter(
    (quote): quote is typeof quote & { symbol: string; quoteType: string; exchange?: string } =>
      "symbol" in quote &&
      typeof quote.symbol === "string" &&
      quote.symbol.length > 0 &&
      "quoteType" in quote &&
      (quote.quoteType === "EQUITY" || quote.quoteType === "ETF")
  );
  const best =
    candidates.find((q) => q.exchange && US_EXCHANGES.has(q.exchange)) ?? candidates[0];
  if (!best) throw new SymbolNotFoundError(rawQuery);
  return best.symbol;
}
