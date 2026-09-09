import { NextResponse, type NextRequest } from "next/server";
import { hashToken } from "@/lib/auth/tokens";
import { createSession, applySessionCookies } from "@/lib/auth/session";
import * as authTokens from "@/lib/db/queries/authTokens";
import * as users from "@/lib/db/queries/users";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const token = typeof body?.token === "string" ? body.token : "";
  if (!token) {
    return NextResponse.json({ error: "Invalid or expired token" }, { status: 400 });
  }

  const tokenRow = await authTokens.getValidByHash(hashToken(token), "email_verify");
  if (!tokenRow) {
    return NextResponse.json({ error: "Invalid or expired token" }, { status: 400 });
  }

  users.markEmailVerified(tokenRow.userId);
  authTokens.consume(tokenRow.id);

  const user = await users.getById(tokenRow.userId);
  const { raw } = await createSession(tokenRow.userId);
  const response = NextResponse.json({ email: user?.email ?? null });
  applySessionCookies(response, raw);
  return response;
}
