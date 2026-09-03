export interface Headline {
  headline: string;
  summary: string;
  source: string;
  url: string;
  publishedAt: string;
}

function isEnabled(): boolean {
  return Boolean(process.env.FINNHUB_API_KEY);
}

function toDateParam(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** Company headlines from Finnhub for the movement-insight LLM prompt. */
export async function fetchFinnhubCompanyNews(
  symbol: string,
  lookbackDays = 5,
  maxArticles = 6
): Promise<Headline[]> {
  if (!isEnabled()) return [];

  const to = new Date();
  const from = new Date(to.getTime() - lookbackDays * 24 * 60 * 60 * 1000);

  const url = new URL("https://finnhub.io/api/v1/company-news");
  url.searchParams.set("symbol", symbol);
  url.searchParams.set("from", toDateParam(from));
  url.searchParams.set("to", toDateParam(to));
  url.searchParams.set("token", process.env.FINNHUB_API_KEY!);

  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!response.ok) return [];
    const data: Array<{
      datetime: number;
      headline: string;
      url: string;
      summary: string;
      source: string;
    }> = await response.json();

    const seen = new Set<string>();
    const deduped: Headline[] = [];
    for (const item of data) {
      const key = `${item.headline.toLowerCase()}|${item.url.toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push({
        headline: item.headline,
        summary: item.summary,
        source: item.source,
        url: item.url,
        publishedAt: new Date(item.datetime * 1000).toISOString(),
      });
    }

    return deduped
      .sort((a, b) => (a.publishedAt < b.publishedAt ? 1 : -1))
      .slice(0, maxArticles);
  } catch (error) {
    console.error(`[finnhub] company-news failed for ${symbol}`, error);
    return [];
  }
}
