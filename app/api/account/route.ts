import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import { clearSessionCookies } from "@/lib/auth/session";
import * as users from "@/lib/db/queries/users";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const user = await users.getById(userId);
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  return NextResponse.json({
    email: user.email,
    emailVerified: Boolean(user.emailVerifiedAt),
    hasPassword: Boolean(user.passwordHash),
    hasGoogle: Boolean(user.googleSub),
  });
}

export async function DELETE(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  users.deleteUser(userId); // cascades sessions/auth_token/watchlist_item/search_history_item
  const response = new NextResponse(null, { status: 204 });
  clearSessionCookies(response);
  return response;
}
