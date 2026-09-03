import { and, desc, eq, gte } from "drizzle-orm";
import { db } from "../client";
import { sentimentHistory } from "../schema";

export type SentimentLabel = "positive" | "negative" | "neutral";

export interface NewSentimentRecord {
  ticker: string;
  articleTitle: string;
  articleLink: string | null;
  articleSource: string | null;
  articlePublishedAt: string | null;
  sentiment: SentimentLabel;
}

/** Mirrors the old app's dedupe rule: only skip when the article has a link. */
export async function findExisting(
  ticker: string,
  articleLink: string | null
) {
  if (!articleLink) return undefined;
  return db.query.sentimentHistory.findFirst({
    where: and(
      eq(sentimentHistory.ticker, ticker),
      eq(sentimentHistory.articleLink, articleLink)
    ),
  });
}

export function insert(record: NewSentimentRecord) {
  return db.insert(sentimentHistory).values(record).returning().get();
}

/** Used for the 90-day sentiment-vs-price overlay chart. */
export async function listSince(ticker: string, sinceIso: string) {
  return db.query.sentimentHistory.findMany({
    where: and(
      eq(sentimentHistory.ticker, ticker),
      gte(sentimentHistory.analyzedAt, sinceIso)
    ),
    orderBy: desc(sentimentHistory.analyzedAt),
  });
}
