import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import * as users from "@/lib/db/queries/users";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ loggedIn: false, email: null });

  const user = await users.getById(userId);
  if (!user) return NextResponse.json({ loggedIn: false, email: null });

  return NextResponse.json({ loggedIn: true, email: user.email });
}
