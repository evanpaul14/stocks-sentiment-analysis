import { createHash, randomBytes } from "node:crypto";

/**
 * Generates an opaque token with 256 bits of entropy. The raw value is what
 * goes into a cookie or an emailed link; only its hash is ever persisted, so
 * a database dump alone can't be replayed to hijack a session or a pending
 * verification/reset link.
 */
export function generateOpaqueToken(): { raw: string; hash: string } {
  const raw = randomBytes(32).toString("base64url");
  return { raw, hash: hashToken(raw) };
}

export function hashToken(raw: string): string {
  return createHash("sha256").update(raw).digest("hex");
}
