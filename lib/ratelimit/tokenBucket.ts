interface Bucket {
  tokens: number;
  lastRefill: number;
}

/**
 * In-memory per-key token bucket. Acceptable tradeoff for a single-VPS
 * process: state resets on restart, but this is abuse protection, not
 * billing-critical, and avoids standing up Redis for one process. If this
 * ever moves to multiple instances behind a load balancer, swap for a
 * shared store (Redis) deliberately — don't just scale instances silently.
 */
class TokenBucketLimiter {
  private readonly buckets = new Map<string, Bucket>();

  constructor(
    private readonly limit: number,
    private readonly windowMs: number
  ) {
    setInterval(() => this.sweep(), 10 * 60 * 1000).unref();
  }

  /** Returns true if the request is allowed, false if it should be rejected. */
  consume(key: string): boolean {
    const now = Date.now();
    const refillRate = this.limit / this.windowMs;
    const bucket = this.buckets.get(key) ?? { tokens: this.limit, lastRefill: now };

    const elapsed = now - bucket.lastRefill;
    bucket.tokens = Math.min(this.limit, bucket.tokens + elapsed * refillRate);
    bucket.lastRefill = now;

    if (bucket.tokens < 1) {
      this.buckets.set(key, bucket);
      return false;
    }

    bucket.tokens -= 1;
    this.buckets.set(key, bucket);
    return true;
  }

  private sweep() {
    const now = Date.now();
    for (const [key, bucket] of this.buckets) {
      if (now - bucket.lastRefill > this.windowMs * 2) {
        this.buckets.delete(key);
      }
    }
  }
}

const limiters = new Map<string, TokenBucketLimiter>();

/** Gets (or lazily creates) the shared limiter for a given route name + quota. */
export function getLimiter(routeName: string, limit: number, windowMs: number): TokenBucketLimiter {
  const cacheKey = `${routeName}:${limit}:${windowMs}`;
  let limiter = limiters.get(cacheKey);
  if (!limiter) {
    limiter = new TokenBucketLimiter(limit, windowMs);
    limiters.set(cacheKey, limiter);
  }
  return limiter;
}
