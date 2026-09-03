import { eq } from "drizzle-orm";
import { db } from "../client";
import { articleImage } from "../schema";

export interface NewArticleImage {
  key: string;
  unsplashId: string | null;
  url: string;
  thumbnailUrl: string | null;
  downloadLocation: string | null;
  attributionJson: string | null;
}

export async function getByKey(key: string) {
  return db.query.articleImage.findFirst({
    where: eq(articleImage.key, key),
  });
}

/** Race-safe: a concurrent insert on the same key is fine, we just re-read after. */
export async function insertIfAbsent(record: NewArticleImage) {
  db.insert(articleImage).values(record).onConflictDoNothing().run();
  return getByKey(record.key);
}
