import { and, desc, eq } from "drizzle-orm";
import { db } from "../client";
import { watchlistItem } from "../schema";

export function listForUser(userId: number) {
  return db.query.watchlistItem.findMany({
    where: eq(watchlistItem.userId, userId),
    orderBy: desc(watchlistItem.addedAt),
  });
}

export function add(userId: number, symbol: string, companyName: string) {
  return db
    .insert(watchlistItem)
    .values({ userId, symbol, companyName })
    .onConflictDoNothing({
      target: [watchlistItem.userId, watchlistItem.symbol],
    })
    .run();
}

export function remove(userId: number, symbol: string) {
  return db
    .delete(watchlistItem)
    .where(
      and(eq(watchlistItem.userId, userId), eq(watchlistItem.symbol, symbol))
    )
    .run();
}

export interface LocalWatchlistEntry {
  symbol: string;
  companyName: string;
}

/** Idempotent bulk upsert used for the one-time localStorage->account merge on login. */
export function mergeFromLocal(
  userId: number,
  entries: LocalWatchlistEntry[]
) {
  for (const entry of entries) {
    add(userId, entry.symbol, entry.companyName);
  }
}
