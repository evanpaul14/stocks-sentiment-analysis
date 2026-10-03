/** Free stock lookups an anonymous visitor (keyed by IP) gets per window. */
export const ANON_SEARCH_LIMIT = 15;
const WINDOW_MS = 24 * 60 * 60 * 1000;

// ip -> (symbol -> time first viewed). Counting distinct symbols means
// refreshing or revisiting a stock you already looked up doesn't burn quota.
// In-memory like the rest of lib/ratelimit: resets on restart, single process.
const usage = new Map<string, Map<string, number>>();

setInterval(() => {
  const cutoff = Date.now() - WINDOW_MS;
  for (const [ip, symbols] of usage) {
    for (const [symbol, at] of symbols) {
      if (at < cutoff) symbols.delete(symbol);
    }
    if (symbols.size === 0) usage.delete(ip);
  }
}, 10 * 60 * 1000).unref();

export interface AnonSearchStatus {
  used: number;
  limit: number;
  limited: boolean;
}

/** Records a lookup of `symbol` by `ip` and reports whether it's allowed. */
export function recordAnonSearch(ip: string, symbol: string): AnonSearchStatus {
  const now = Date.now();
  let symbols = usage.get(ip);
  if (!symbols) {
    symbols = new Map();
    usage.set(ip, symbols);
  }
  for (const [s, at] of symbols) {
    if (now - at > WINDOW_MS) symbols.delete(s);
  }

  const key = symbol.toUpperCase();
  if (!symbols.has(key)) {
    if (symbols.size >= ANON_SEARCH_LIMIT) {
      return { used: symbols.size, limit: ANON_SEARCH_LIMIT, limited: true };
    }
    symbols.set(key, now);
  }
  return { used: symbols.size, limit: ANON_SEARCH_LIMIT, limited: false };
}
