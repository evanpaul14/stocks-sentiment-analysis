import type { NextRequest, NextResponse } from "next/server";
import * as sessions from "@/lib/db/queries/sessions";
import { generateOpaqueToken, hashToken } from "./tokens";

export const SESSION_COOKIE = "ssa_session";
/** Non-httpOnly flag cookie so the client can render logged-in/out state without a round trip. Carries no session material. */
export const LOGGED_IN_COOKIE = "ssa_logged_in";

const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30; // 30 days
const STALE_TOUCH_THRESHOLD_MS = 60 * 60 * 1000; // bump lastSeenAt at most once/hour

export async function createSession(
  userId: number
): Promise<{ raw: string; expiresAt: string }> {
  const { raw, hash } = generateOpaqueToken();
  const expiresAt = new Date(
    Date.now() + SESSION_MAX_AGE_SECONDS * 1000
  ).toISOString();
  sessions.insert(userId, hash, expiresAt);
  return { raw, expiresAt };
}

/** Must be Lax, not Strict — the Google OAuth callback is a top-level GET redirect that needs this cookie. */
export function applySessionCookies(response: NextResponse, raw: string) {
  const isProd = process.env.NODE_ENV === "production";
  response.cookies.set(SESSION_COOKIE, raw, {
    httpOnly: true,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_MAX_AGE_SECONDS,
  });
  response.cookies.set(LOGGED_IN_COOKIE, "1", {
    httpOnly: false,
    secure: isProd,
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_MAX_AGE_SECONDS,
  });
}

export function clearSessionCookies(response: NextResponse) {
  response.cookies.delete(SESSION_COOKIE);
  response.cookies.delete(LOGGED_IN_COOKIE);
}

export interface ResolvedSession {
  sessionId: number;
  userId: number;
}

/** Looks up the session for the current request's cookie, if any, valid and unexpired. */
export async function resolveSession(
  request: NextRequest
): Promise<ResolvedSession | null> {
  const raw = request.cookies.get(SESSION_COOKIE)?.value;
  if (!raw) return null;

  const row = await sessions.getByTokenHash(hashToken(raw));
  if (!row) return null;
  if (row.revokedAt) return null;
  if (new Date(row.expiresAt).getTime() <= Date.now()) return null;

  if (Date.now() - new Date(row.lastSeenAt).getTime() > STALE_TOUCH_THRESHOLD_MS) {
    sessions.touchLastSeen(row.id);
  }

  return { sessionId: row.id, userId: row.userId };
}

export function revokeAllSessionsForUser(userId: number): void {
  sessions.revokeAllForUser(userId);
}

export async function revokeSessionForRequest(
  request: NextRequest
): Promise<void> {
  const raw = request.cookies.get(SESSION_COOKIE)?.value;
  if (!raw) return;

  const row = await sessions.getByTokenHash(hashToken(raw));
  if (row) sessions.revoke(row.id);
}
