const WATCHLIST_KEY = "ssa_watchlist_v1";

export interface WatchlistEntry {
  symbol: string;
  companyName: string;
  lastPrice: number | null;
  lastChangePercent: number | null;
  addedAt: string;
}

let cachedRaw: string | null | undefined;
let cachedEntries: WatchlistEntry[] = [];

export function readWatchlist(): WatchlistEntry[] {
  try {
    const raw = localStorage.getItem(WATCHLIST_KEY);
    // Return the same array reference when the underlying value hasn't
    // changed — required for useSyncExternalStore's getSnapshot to avoid
    // re-rendering (and looping) on every call.
    if (raw === cachedRaw) return cachedEntries;

    cachedRaw = raw;
    if (!raw) {
      cachedEntries = [];
    } else {
      const parsed = JSON.parse(raw);
      cachedEntries = Array.isArray(parsed) ? parsed : [];
    }
    return cachedEntries;
  } catch {
    return [];
  }
}

export function subscribeToWatchlist(callback: () => void): () => void {
  window.addEventListener("ssa-watchlist-changed", callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener("ssa-watchlist-changed", callback);
    window.removeEventListener("storage", callback);
  };
}

function writeWatchlist(entries: WatchlistEntry[]) {
  try {
    localStorage.setItem(WATCHLIST_KEY, JSON.stringify(entries));
    window.dispatchEvent(new Event("ssa-watchlist-changed"));
  } catch {
    // localStorage unavailable — degrade silently
  }
}

export function isInWatchlist(symbol: string): boolean {
  return readWatchlist().some((e) => e.symbol === symbol);
}

export function addToWatchlist(entry: Omit<WatchlistEntry, "addedAt">) {
  const current = readWatchlist();
  if (current.some((e) => e.symbol === entry.symbol)) return;
  writeWatchlist([...current, { ...entry, addedAt: new Date().toISOString() }]);
}

export function removeFromWatchlist(symbol: string) {
  writeWatchlist(readWatchlist().filter((e) => e.symbol !== symbol));
}

export function updateWatchlistPrices(
  prices: Map<string, { price: number | null; changePercent: number | null }>
) {
  const current = readWatchlist();
  if (current.length === 0) return;
  writeWatchlist(
    current.map((entry) => {
      const update = prices.get(entry.symbol);
      if (!update) return entry;
      return {
        ...entry,
        lastPrice: update.price ?? entry.lastPrice,
        lastChangePercent: update.changePercent ?? entry.lastChangePercent,
      };
    })
  );
}
