import { and, eq, isNull, lt, sql } from "drizzle-orm";
import { db } from "../client";
import { session } from "../schema";

export function insert(userId: number, tokenHash: string, expiresAt: string) {
  return db
    .insert(session)
    .values({ userId, tokenHash, expiresAt })
    .returning()
    .get();
}

export function getByTokenHash(tokenHash: string) {
  return db.query.session.findFirst({
    where: eq(session.tokenHash, tokenHash),
  });
}

export function touchLastSeen(id: number) {
  return db
    .update(session)
    .set({ lastSeenAt: sql`(current_timestamp)` })
    .where(eq(session.id, id))
    .run();
}

export function revoke(id: number) {
  return db
    .update(session)
    .set({ revokedAt: sql`(current_timestamp)` })
    .where(eq(session.id, id))
    .run();
}

export function revokeAllForUser(userId: number) {
  return db
    .update(session)
    .set({ revokedAt: sql`(current_timestamp)` })
    .where(and(eq(session.userId, userId), isNull(session.revokedAt)))
    .run();
}

/** Opportunistic cleanup — safe to call from a low-traffic path or cron. */
export function deleteExpired() {
  return db
    .delete(session)
    .where(lt(session.expiresAt, sql`(current_timestamp)`))
    .run();
}
