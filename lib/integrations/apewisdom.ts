const APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks";

export interface RedditTrendingItem {
  rank: number;
  ticker: string;
  name: string;
  mentions: number;
  mentionsChangePercent: number;
  tag: { type: "trending" } | { type: "up-spots"; spots: number } | { type: "none" };
}

interface ApeWisdomResponse {
  results: Array<{
    rank: number | null;
    ticker?: string;
    name: string;
    mentions: number;
    mentions_24h_ago: number;
    rank_24h_ago: number;
  }>;
}

/** Reddit mention-count trending, via ApeWisdom. Degrades to [] on failure. */
export async function fetchRedditTrending(limit = 10): Promise<RedditTrendingItem[]> {
  try {
    const response = await fetch(APEWISDOM_URL, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return [];
    const data: ApeWisdomResponse = await response.json();

    return data.results
      .filter((r) => r.rank != null && r.ticker)
      .sort((a, b) => (a.rank as number) - (b.rank as number))
      .slice(0, limit)
      .map((r) => {
        const pctIncrease =
          r.mentions_24h_ago > 0
            ? ((r.mentions - r.mentions_24h_ago) / r.mentions_24h_ago) * 100
            : 0;
        const rank = r.rank as number;
        const tag: RedditTrendingItem["tag"] =
          r.mentions_24h_ago > 0 && pctIncrease > 50
            ? { type: "trending" }
            : rank < r.rank_24h_ago - 5
              ? { type: "up-spots", spots: r.rank_24h_ago - rank }
              : { type: "none" };

        return {
          rank,
          ticker: r.ticker as string,
          name: r.name,
          mentions: r.mentions,
          mentionsChangePercent: Number(pctIncrease.toFixed(2)),
          tag,
        };
      });
  } catch (error) {
    console.error("[apewisdom] fetch failed", error);
    return [];
  }
}
