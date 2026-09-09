import { NextResponse, type NextRequest } from "next/server";
import { hashPassword } from "@/lib/auth/password";
import { isPasswordValid, PASSWORD_REQUIREMENTS_MESSAGE } from "@/lib/auth/passwordPolicy";
import { hashToken } from "@/lib/auth/tokens";
import { revokeAllSessionsForUser } from "@/lib/auth/session";
import * as authTokens from "@/lib/db/queries/authTokens";
import * as users from "@/lib/db/queries/users";

export async function POST(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const token = typeof body?.token === "string" ? body.token : "";
  const newPassword = typeof body?.newPassword === "string" ? body.newPassword : "";

  if (!token || !isPasswordValid(newPassword)) {
    return NextResponse.json({ error: PASSWORD_REQUIREMENTS_MESSAGE }, { status: 400 });
  }

  const tokenRow = await authTokens.getValidByHash(hashToken(token), "password_reset");
  if (!tokenRow) {
    return NextResponse.json({ error: "Invalid or expired token" }, { status: 400 });
  }

  const { hash, salt } = hashPassword(newPassword);
  users.updatePassword(tokenRow.userId, hash, salt);
  authTokens.consume(tokenRow.id);
  // A password reset should kill any session an attacker may already hold.
  revokeAllSessionsForUser(tokenRow.userId);

  return new NextResponse(null, { status: 204 });
}
