import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { isValidEmail, isMailgunEnabled, sendVerificationEmail } from "@/lib/integrations/mailgun";
import { hashPassword } from "@/lib/auth/password";
import { isPasswordValid, PASSWORD_REQUIREMENTS_MESSAGE } from "@/lib/auth/passwordPolicy";
import { generateOpaqueToken } from "@/lib/auth/tokens";
import * as users from "@/lib/db/queries/users";
import * as authTokens from "@/lib/db/queries/authTokens";

const VERIFY_TOKEN_TTL_MS = 24 * 60 * 60 * 1000;

async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  const password = typeof body?.password === "string" ? body.password : "";

  if (!isValidEmail(email)) {
    return NextResponse.json({ error: "Invalid email address" }, { status: 400 });
  }
  if (!isPasswordValid(password)) {
    return NextResponse.json({ error: PASSWORD_REQUIREMENTS_MESSAGE }, { status: 400 });
  }

  const { hash, salt } = hashPassword(password);
  const created = await users.insertWithPassword(email, hash, salt);
  if (!created) {
    return NextResponse.json({ error: "Email already in use" }, { status: 409 });
  }

  if (isMailgunEnabled()) {
    const { raw, hash: tokenHash } = generateOpaqueToken();
    const expiresAt = new Date(Date.now() + VERIFY_TOKEN_TTL_MS).toISOString();
    authTokens.insert(created.id, "email_verify", tokenHash, expiresAt);

    const verifyUrl = `${process.env.SITE_BASE_URL ?? ""}/verify-email?token=${raw}`;
    try {
      await sendVerificationEmail(email, verifyUrl);
    } catch (error) {
      console.error("[api/auth/signup] verification email failed", error);
    }
  }

  return new NextResponse(null, { status: 204 });
}

export const POST = withRateLimit(
  { routeName: "auth-signup", limit: 5, windowMs: 60 * 60_000 },
  handler
);
