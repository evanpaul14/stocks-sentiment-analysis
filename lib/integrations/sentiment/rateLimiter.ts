/**
 * In-process sliding-window rate limiter. Reusable across any client that
 * needs to stay under a calls-per-window quota (currently just Gemini).
 */
export class SlidingWindowRateLimiter {
  private readonly callTimestamps: number[] = [];

  constructor(
    private readonly maxCallsPerWindow: number,
    private readonly windowMs: number
  ) {}

  async waitForSlot(): Promise<void> {
    if (this.maxCallsPerWindow <= 0) return; // unlimited

    for (;;) {
      const now = Date.now();
      while (
        this.callTimestamps.length > 0 &&
        now - this.callTimestamps[0] > this.windowMs
      ) {
        this.callTimestamps.shift();
      }

      if (this.callTimestamps.length < this.maxCallsPerWindow) {
        this.callTimestamps.push(now);
        return;
      }

      const oldest = this.callTimestamps[0];
      const waitMs = Math.max(this.windowMs - (now - oldest), 100);
      await sleep(waitMs);
    }
  }
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
