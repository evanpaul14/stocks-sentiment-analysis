import { stripHtml, type NewsArticle } from "./types";

interface AlpacaNewsResponse {
  news?: Array<{
    headline?: string;
    summary?: string;
    url?: string;
    source?: string;
    created_at?: string;
  }>;
}

function credentials(): { id: string; secret: string } | null {
  const id = process.env.ALPACA_API_KEY_ID || process.env.ALPACA_API_KEY;
  const secret = process.env.ALPACA_API_SECRET_KEY || process.env.ALPACA_SECRET_KEY;
  return id && secret ? { id, secret } : null;
}

/** Alpaca's news endpoint is keyed by ticker, so free-text queries can't use it. */
export function isTickerLike(query: string): boolean {
  return /^[A-Za-z]{1,5}([.-][A-Za-z])?$/.test(query);
}

/** Alpaca tags dual-class companies under one share class (Alphabet news is all `GOOG`), so query both. */
const SHARE_CLASS_SYMBOLS: Record<string, string> = {
  GOOGL: "GOOG,GOOGL",
  GOOG: "GOOG,GOOGL",
};

/** Last-resort fallback: tightly on-topic, with summaries, but little publisher variety. */
export async function fetchAlpacaNews(symbol: string, limit: number): Promise<NewsArticle[]> {
  const creds = credentials();
  if (!creds || !isTickerLike(symbol)) return [];

  const url = new URL("https://data.alpaca.markets/v1beta1/news");
  const upper = symbol.toUpperCase();
  url.searchParams.set("symbols", SHARE_CLASS_SYMBOLS[upper] ?? upper);
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("sort", "desc");

  const response = await fetch(url, {
    headers: { "Apca-Api-Key-Id": creds.id, "Apca-Api-Secret-Key": creds.secret },
    signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) throw new Error(`Alpaca news responded ${response.status}`);

  const data: AlpacaNewsResponse = await response.json();
  return (data.news ?? []).flatMap((item) =>
    item.headline && item.url
      ? [
          {
            title: item.headline,
            description: item.summary ? stripHtml(item.summary) : "",
            link: item.url,
            publishedAt: item.created_at ?? null,
            source: item.source || "Alpaca",
          },
        ]
      : []
  );
}
