import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { isMailgunEnabled, sendVerificationEmail } from "@/lib/integrations/mailgun";
import { generateOpaqueToken } from "@/lib/auth/tokens";
import * as users from "@/lib/db/queries/users";
import * as authTokens from "@/lib/db/queries/authTokens";

const VERIFY_TOKEN_TTL_MS = 24 * 60 * 60 * 1000;

async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";

  // Always respond 204 regardless of whether the address exists or is
  // already verified, to avoid letting this endpoint be used to enumerate
  // registered accounts.
  if (email && isMailgunEnabled()) {
    const user = await users.getByEmail(email);
    if (user && !user.emailVerifiedAt) {
      authTokens.deleteAllForUserAndPurpose(user.id, "email_verify");
      const { raw, hash } = generateOpaqueToken();
      const expiresAt = new Date(Date.now() + VERIFY_TOKEN_TTL_MS).toISOString();
      authTokens.insert(user.id, "email_verify", hash, expiresAt);

      const verifyUrl = `${process.env.SITE_BASE_URL ?? ""}/verify-email?token=${raw}`;
      try {
        await sendVerificationEmail(email, verifyUrl);
      } catch (error) {
        console.error("[api/auth/resend-verification] failed", error);
      }
    }
  }

  return new NextResponse(null, { status: 204 });
}

export const POST = withRateLimit(
  { routeName: "auth-resend-verification", limit: 3, windowMs: 60 * 60_000 },
  handler
);
