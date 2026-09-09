import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import * as watchlistItems from "@/lib/db/queries/watchlistItems";

export const dynamic = "force-dynamic";

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const { symbol } = await context.params;
  watchlistItems.remove(userId, symbol.toUpperCase());
  return new NextResponse(null, { status: 204 });
}
