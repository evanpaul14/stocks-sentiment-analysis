import { eq } from "drizzle-orm";
import { db } from "../client";
import { sentimentPageCache } from "../schema";

export interface SentimentPageCacheRecord {
  slug: string;
  ticker: string;
  sectionsJson: string;
  priceJson: string | null;
  sentimentJson: string | null;
  expiresAt: string;
}

export async function getBySlug(slug: string) {
  return db.query.sentimentPageCache.findFirst({
    where: eq(sentimentPageCache.slug, slug),
  });
}

/** Slug + last-refresh timestamp for every cached page — used to set sitemap `lastmod`. */
export async function listSlugsAndGeneratedAt() {
  return db.query.sentimentPageCache.findMany({
    columns: { slug: true, generatedAt: true },
  });
}

export function isFresh(row: { expiresAt: string }) {
  return new Date(row.expiresAt).getTime() > Date.now();
}

export function upsert(record: SentimentPageCacheRecord) {
  const now = new Date().toISOString();
  return db
    .insert(sentimentPageCache)
    .values({ ...record, generatedAt: now, firstGeneratedAt: now })
    .onConflictDoUpdate({
      target: sentimentPageCache.slug,
      set: {
        ticker: record.ticker,
        sectionsJson: record.sectionsJson,
        priceJson: record.priceJson,
        sentimentJson: record.sentimentJson,
        expiresAt: record.expiresAt,
        generatedAt: now,
        // firstGeneratedAt intentionally omitted — it's set once on insert
        // and must stay stable across every subsequent 24h refresh.
      },
    })
    .returning()
    .get();
}
