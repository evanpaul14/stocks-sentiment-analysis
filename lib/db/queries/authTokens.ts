import { and, eq, isNull, gt, sql } from "drizzle-orm";
import { db } from "../client";
import { authToken } from "../schema";

export type AuthTokenPurpose = "email_verify" | "password_reset";

export function insert(
  userId: number,
  purpose: AuthTokenPurpose,
  tokenHash: string,
  expiresAt: string
) {
  return db
    .insert(authToken)
    .values({ userId, purpose, tokenHash, expiresAt })
    .returning()
    .get();
}

/** Only returns a token that hasn't been consumed and hasn't expired. */
export function getValidByHash(tokenHash: string, purpose: AuthTokenPurpose) {
  return db.query.authToken.findFirst({
    where: and(
      eq(authToken.tokenHash, tokenHash),
      eq(authToken.purpose, purpose),
      isNull(authToken.consumedAt),
      gt(authToken.expiresAt, sql`(current_timestamp)`)
    ),
  });
}

export function consume(id: number) {
  return db
    .update(authToken)
    .set({ consumedAt: sql`(current_timestamp)` })
    .where(eq(authToken.id, id))
    .run();
}

/** Invalidate any previously issued tokens so only one link is ever valid at a time. */
export function deleteAllForUserAndPurpose(
  userId: number,
  purpose: AuthTokenPurpose
) {
  return db
    .delete(authToken)
    .where(and(eq(authToken.userId, userId), eq(authToken.purpose, purpose)))
    .run();
}
