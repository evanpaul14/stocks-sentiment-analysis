import type { NextRequest } from "next/server";
import { resolveSession } from "./session";

/** Thin composition of resolveSession, used to gate routes the way isAuthorizedAdminRequest gates admin ones. */
export async function getCurrentUserId(
  request: NextRequest
): Promise<number | null> {
  const session = await resolveSession(request);
  return session?.userId ?? null;
}
