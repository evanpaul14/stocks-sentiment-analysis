/**
 * StockTwits has no public auth. A live check (2026-08) confirmed plain
 * fetch with a realistic browser User-Agent reaches these endpoints without
 * a Cloudflare JS challenge — no headless-browser workaround needed. If
 * that ever changes, this is the one place to swap in a challenge-solving
 * strategy (puppeteer-extra+stealth, or a scraping proxy).
 */
const USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

export async function stocktwitsFetch(
  url: string,
  params?: Record<string, string | number | undefined>
): Promise<unknown> {
  const target = new URL(url);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) target.searchParams.set(key, String(value));
    }
  }

  const response = await fetch(target, {
    headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
    signal: AbortSignal.timeout(10_000),
  });

  if (!response.ok) {
    throw new Error(`StockTwits request failed: ${response.status} ${target}`);
  }

  return response.json();
}
