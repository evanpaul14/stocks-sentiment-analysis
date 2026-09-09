import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { verifyPassword } from "@/lib/auth/password";
import { createSession, applySessionCookies } from "@/lib/auth/session";
import * as users from "@/lib/db/queries/users";

async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
  const password = typeof body?.password === "string" ? body.password : "";

  const invalidCredentials = () =>
    NextResponse.json({ error: "Invalid email or password" }, { status: 401 });

  if (!email || !password) return invalidCredentials();

  const user = await users.getByEmail(email);
  if (!user || !user.passwordHash || !user.passwordSalt) return invalidCredentials();
  if (!verifyPassword(password, user.passwordHash, user.passwordSalt)) {
    return invalidCredentials();
  }

  if (!user.emailVerifiedAt) {
    return NextResponse.json({ error: "email_not_verified" }, { status: 403 });
  }

  const { raw } = await createSession(user.id);
  const response = NextResponse.json({ email: user.email });
  applySessionCookies(response, raw);
  return response;
}

export const POST = withRateLimit(
  { routeName: "auth-login", limit: 10, windowMs: 60 * 60_000 },
  handler
);
