import { eq, sql } from "drizzle-orm";
import { db } from "../client";
import { user } from "../schema";

export function getByEmail(email: string) {
  return db.query.user.findFirst({ where: eq(user.email, email) });
}

export function getById(id: number) {
  return db.query.user.findFirst({ where: eq(user.id, id) });
}

export function getByGoogleSub(googleSub: string) {
  return db.query.user.findFirst({ where: eq(user.googleSub, googleSub) });
}

/**
 * Race-safe against a concurrent signup with the same email: the unique
 * index on `email` makes the insert a no-op on conflict, and we report that
 * back as `null` so the caller can respond "email already in use" instead of
 * silently overwriting or double-creating an account.
 */
export async function insertWithPassword(
  email: string,
  passwordHash: string,
  passwordSalt: string
) {
  const result = db
    .insert(user)
    .values({ email, passwordHash, passwordSalt })
    .onConflictDoNothing({ target: user.email })
    .run();
  if (result.changes === 0) return null;
  return getByEmail(email);
}

/** Google already verifies the email, so the new row is marked verified immediately. */
export async function insertWithGoogle(email: string, googleSub: string) {
  const result = db
    .insert(user)
    .values({ email, googleSub, emailVerifiedAt: sql`(current_timestamp)` })
    .onConflictDoNothing({ target: user.email })
    .run();
  if (result.changes === 0) return null;
  return getByEmail(email);
}

export function linkGoogleSub(userId: number, googleSub: string) {
  return db
    .update(user)
    .set({ googleSub })
    .where(eq(user.id, userId))
    .run();
}

export function markEmailVerified(userId: number) {
  return db
    .update(user)
    .set({ emailVerifiedAt: sql`(current_timestamp)` })
    .where(eq(user.id, userId))
    .run();
}

export function updatePassword(
  userId: number,
  passwordHash: string,
  passwordSalt: string
) {
  return db
    .update(user)
    .set({ passwordHash, passwordSalt })
    .where(eq(user.id, userId))
    .run();
}

export function deleteUser(userId: number) {
  return db.delete(user).where(eq(user.id, userId)).run();
}
