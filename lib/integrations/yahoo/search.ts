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

/** Resolves a free-text company/ticker query to a Yahoo Finance symbol. */
export async function resolveSymbol(rawQuery: string): Promise<string> {
  const query = normalizeSearchQuery(rawQuery);
  const results = await yahooFinance.search(query);
  const firstQuote = results.quotes.find(
    (quote): quote is typeof quote & { symbol: string } =>
      "symbol" in quote && typeof quote.symbol === "string" && quote.symbol.length > 0
  );
  if (!firstQuote) throw new SymbolNotFoundError(rawQuery);
  return firstQuote.symbol;
}
