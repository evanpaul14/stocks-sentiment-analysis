import { and, desc, eq } from "drizzle-orm";
import { db } from "../client";
import { searchHistoryItem } from "../schema";

export async function listForUser(userId: number, limit = 8) {
  return db.query.searchHistoryItem.findMany({
    where: eq(searchHistoryItem.userId, userId),
    orderBy: desc(searchHistoryItem.searchedAt),
    limit,
  });
}

/** Delete-then-insert so a re-searched term moves back to the front, matching the old localStorage behavior. */
export async function record(userId: number, query: string, limit = 8) {
  db.delete(searchHistoryItem)
    .where(
      and(
        eq(searchHistoryItem.userId, userId),
        eq(searchHistoryItem.query, query)
      )
    )
    .run();
  db.insert(searchHistoryItem).values({ userId, query }).run();

  // Drizzle's relational query builder can't emit OFFSET without LIMIT, so
  // fetch everything ordered and slice in JS instead.
  const all = await db.query.searchHistoryItem.findMany({
    where: eq(searchHistoryItem.userId, userId),
    orderBy: desc(searchHistoryItem.searchedAt),
  });
  const overflow = all.slice(limit);
  for (const row of overflow) {
    db.delete(searchHistoryItem)
      .where(eq(searchHistoryItem.id, row.id))
      .run();
  }
}

/** Idempotent-enough bulk insert used for the one-time localStorage->account merge on login. */
export async function mergeFromLocal(userId: number, queries: string[]) {
  for (const query of queries) {
    await record(userId, query);
  }
}
