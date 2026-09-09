import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import { createAdminClient } from "@/lib/supabase/admin";
import { db } from "@/lib/db/client";
import { searchHistoryItem, watchlistItem } from "@/lib/db/schema";
import { eq } from "drizzle-orm";

export const dynamic = "force-dynamic";

/**
 * Deletes the caller's own Supabase Auth user (via the service-role admin
 * client — the regular client SDK can't delete accounts) plus their local
 * watchlist/search-history rows, which have no FK to cascade since identity
 * now lives entirely in Supabase, not this database.
 */
export async function DELETE(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const admin = createAdminClient();
  const { error } = await admin.auth.admin.deleteUser(userId);
  if (error) {
    console.error("[api/account] deleteUser failed", error);
    return NextResponse.json({ error: "Failed to delete account" }, { status: 502 });
  }

  db.delete(watchlistItem).where(eq(watchlistItem.userId, userId)).run();
  db.delete(searchHistoryItem).where(eq(searchHistoryItem.userId, userId)).run();

  return new NextResponse(null, { status: 204 });
}
