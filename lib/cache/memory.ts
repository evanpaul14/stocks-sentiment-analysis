interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

const DEFAULT_MAX_ENTRIES = 200;

/**
 * Generic in-memory TTL cache for hot, short-lived data (StockTwits summaries,
 * quote polling, etc). Not shared across processes — fine for a single-VPS
 * deployment; longer-lived data belongs in SQLite instead (see lib/db).
 *
 * Keys are often ticker symbols, which cover a long tail of companies (SEO
 * sentiment pages) that a given key may only ever be requested once for.
 * Expired entries are only reclaimed lazily on a matching `get`, so without a
 * cap the store grows unboundedly with crawl traffic. `maxEntries` bounds it
 * via oldest-first eviction (Map preserves insertion order).
 */
export class TtlCache<T> {
  private readonly store = new Map<string, CacheEntry<T>>();
  private readonly inFlight = new Map<string, Promise<T>>();

  constructor(
    private readonly ttlMs: number,
    private readonly maxEntries: number = DEFAULT_MAX_ENTRIES
  ) {}

  get(key: string): T | undefined {
    const entry = this.store.get(key);
    if (!entry) return undefined;
    if (Date.now() > entry.expiresAt) {
      this.store.delete(key);
      return undefined;
    }
    return entry.value;
  }

  set(key: string, value: T): void {
    this.store.delete(key);
    if (this.store.size >= this.maxEntries) {
      const oldestKey = this.store.keys().next().value;
      if (oldestKey !== undefined) this.store.delete(oldestKey);
    }
    this.store.set(key, { value, expiresAt: Date.now() + this.ttlMs });
  }

  /**
   * Fetches from cache, or computes + caches on miss. Concurrent misses for the
   * same key share one in-flight `compute()` call instead of each firing their
   * own (a "cache stampede") — important since `compute` is typically a paid or
   * rate-limited external API call.
   */
  async getOrCompute(key: string, compute: () => Promise<T>): Promise<T> {
    const cached = this.get(key);
    if (cached !== undefined) return cached;

    const pending = this.inFlight.get(key);
    if (pending) return pending;

    const promise = compute()
      .then((value) => {
        this.set(key, value);
        return value;
      })
      .finally(() => {
        this.inFlight.delete(key);
      });
    this.inFlight.set(key, promise);
    return promise;
  }
}
