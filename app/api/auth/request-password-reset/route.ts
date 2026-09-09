import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { isMailgunEnabled, sendPasswordResetEmail } from "@/lib/integrations/mailgun";
import { generateOpaqueToken } from "@/lib/auth/tokens";
import * as users from "@/lib/db/queries/users";
import * as authTokens from "@/lib/db/queries/authTokens";

const RESET_TOKEN_TTL_MS = 60 * 60 * 1000;

async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";

  // Always respond 204 regardless of whether the address exists, to avoid
  // letting this endpoint be used to enumerate registered accounts.
  if (email && isMailgunEnabled()) {
    const user = await users.getByEmail(email);
    if (user && user.passwordHash) {
      authTokens.deleteAllForUserAndPurpose(user.id, "password_reset");
      const { raw, hash } = generateOpaqueToken();
      const expiresAt = new Date(Date.now() + RESET_TOKEN_TTL_MS).toISOString();
      authTokens.insert(user.id, "password_reset", hash, expiresAt);

      const resetUrl = `${process.env.SITE_BASE_URL ?? ""}/reset-password?token=${raw}`;
      try {
        await sendPasswordResetEmail(email, resetUrl);
      } catch (error) {
        console.error("[api/auth/request-password-reset] failed", error);
      }
    }
  }

  return new NextResponse(null, { status: 204 });
}

export const POST = withRateLimit(
  { routeName: "auth-request-password-reset", limit: 3, windowMs: 60 * 60_000 },
  handler
);
