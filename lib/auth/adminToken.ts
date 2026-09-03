import { timingSafeEqual } from "node:crypto";
import type { NextRequest } from "next/server";

/**
 * Single static bearer token protecting low-traffic admin actions (e.g.
 * manual market-summary regeneration). No accounts system exists in this
 * app — this is intentionally the whole auth story for that one endpoint.
 */
export function isAuthorizedAdminRequest(request: NextRequest): boolean {
  const expected = process.env.ADMIN_API_TOKEN;
  if (!expected) return false;

  const header = request.headers.get("authorization");
  const provided = header?.startsWith("Bearer ") ? header.slice(7) : null;
  if (!provided) return false;

  const expectedBuf = Buffer.from(expected);
  const providedBuf = Buffer.from(provided);
  if (expectedBuf.length !== providedBuf.length) return false;

  return timingSafeEqual(expectedBuf, providedBuf);
}
