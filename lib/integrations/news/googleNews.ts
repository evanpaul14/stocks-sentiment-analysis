import Parser from "rss-parser";

export interface NewsArticle {
  title: string;
  description: string;
  link: string;
  publishedAt: string | null;
  source: string;
}

interface GoogleNewsItem {
  title?: string;
  link?: string;
  pubDate?: string;
  contentSnippet?: string;
  source?: { _?: string; $?: { url?: string } } | string;
}

const parser = new Parser<unknown, GoogleNewsItem>({
  customFields: {
    item: [["source", "source", { keepArray: false }]],
  },
});

function stripHtml(html: string): string {
  return html
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function extractSourceName(source: GoogleNewsItem["source"]): string {
  if (!source) return "Unknown";
  if (typeof source === "string") return stripHtml(source) || "Unknown";
  return source._ ?? "Unknown";
}

async function fetchGoogleNews(query: string): Promise<GoogleNewsItem[]> {
  const url = `https://news.google.com/rss/search?q=${encodeURIComponent(
    query
  )}&hl=en-US&gl=US&ceid=US:en`;
  const feed = await parser.parseURL(url);
  return feed.items ?? [];
}

function toArticle(item: GoogleNewsItem): NewsArticle | null {
  if (!item.title || !item.link) return null;
  return {
    title: item.title,
    description: item.contentSnippet ? stripHtml(item.contentSnippet) : "",
    link: item.link,
    publishedAt: item.pubDate ?? null,
    source: extractSourceName(item.source),
  };
}

/** Per-ticker news, used by /api/search (sentiment excluded, kept fast). */
export async function getNewsArticles(
  symbol: string,
  limit = 10
): Promise<NewsArticle[]> {
  try {
    const items = await fetchGoogleNews(`${symbol} stock`);
    return items
      .slice(0, limit)
      .map(toArticle)
      .filter((a): a is NewsArticle => a !== null);
  } catch (error) {
    console.error(`[googleNews] getNewsArticles(${symbol}) failed`, error);
    return [];
  }
}

const MARKET_DIGEST_QUERIES = [
  "stock market today",
  "wall street wrap",
  "us stocks closing bell",
];

/** Market-wide headline digest for the daily market summary. */
export async function getMarketNewsDigest(
  limit = 8
): Promise<NewsArticle[]> {
  const seenTitles = new Set<string>();
  const collected: NewsArticle[] = [];

  for (const query of MARKET_DIGEST_QUERIES) {
    if (collected.length >= limit) break;
    try {
      const items = await fetchGoogleNews(query);
      for (const item of items) {
        if (collected.length >= limit) break;
        const article = toArticle(item);
        if (!article) continue;
        const key = article.title.toLowerCase();
        if (seenTitles.has(key)) continue;
        seenTitles.add(key);
        collected.push(article);
      }
    } catch (error) {
      console.error(`[googleNews] getMarketNewsDigest query "${query}" failed`, error);
    }
  }

  return collected;
}
