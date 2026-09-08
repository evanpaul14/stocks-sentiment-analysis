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

/**
 * Concurrent requests for the same ticker+article can both pass findExisting
 * before either finishes inserting, so this must tolerate losing the race
 * against the unique index instead of throwing SQLITE_CONSTRAINT_UNIQUE.
 */
export async function insert(record: NewSentimentRecord) {
  const inserted = db
    .insert(sentimentHistory)
    .values(record)
    .onConflictDoNothing({
      target: [sentimentHistory.ticker, sentimentHistory.articleLink],
    })
    .returning()
    .get();

  if (inserted) return inserted;

  return findExisting(record.ticker, record.articleLink);
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
