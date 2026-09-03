import { desc, eq } from "drizzle-orm";
import { db } from "../client";
import { marketWrap } from "../schema";

export interface NewMarketWrap {
  date: string;
  slug: string;
  title: string;
  body: string;
  indexSnapshotJson: string;
  headlinesJson: string;
  imageUrl?: string | null;
  imageThumbnailUrl?: string | null;
  imagePhotographerName?: string | null;
  imagePhotographerProfileUrl?: string | null;
}

export async function getLatest() {
  return db.query.marketWrap.findFirst({
    orderBy: desc(marketWrap.date),
  });
}

export async function getByDate(date: string) {
  return db.query.marketWrap.findFirst({
    where: eq(marketWrap.date, date),
  });
}

export async function getBySlug(slug: string) {
  return db.query.marketWrap.findFirst({
    where: eq(marketWrap.slug, slug),
  });
}

export async function listArchive(limit = 10) {
  return db.query.marketWrap.findMany({
    orderBy: desc(marketWrap.date),
    limit,
  });
}

/** Insert, or overwrite the same-day row if a regenerate ran twice (date is UNIQUE). */
export function upsertByDate(record: NewMarketWrap) {
  return db
    .insert(marketWrap)
    .values(record)
    .onConflictDoUpdate({
      target: marketWrap.date,
      set: {
        slug: record.slug,
        title: record.title,
        body: record.body,
        indexSnapshotJson: record.indexSnapshotJson,
        headlinesJson: record.headlinesJson,
        imageUrl: record.imageUrl,
        imageThumbnailUrl: record.imageThumbnailUrl,
        imagePhotographerName: record.imagePhotographerName,
        imagePhotographerProfileUrl: record.imagePhotographerProfileUrl,
      },
    })
    .returning()
    .get();
}
