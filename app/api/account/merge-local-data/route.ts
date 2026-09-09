import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import * as watchlistItems from "@/lib/db/queries/watchlistItems";
import * as searchHistoryItems from "@/lib/db/queries/searchHistoryItems";

export const dynamic = "force-dynamic";

interface LocalWatchlistEntry {
  symbol: string;
  companyName: string;
}

/** One-time merge of a device's localStorage watchlist/search history into the account on login. Idempotent — safe to retry. */
export async function POST(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const body = await request.json().catch(() => null);

  const watchlist: LocalWatchlistEntry[] = Array.isArray(body?.watchlist)
    ? body.watchlist.filter(
        (e: unknown): e is LocalWatchlistEntry =>
          typeof e === "object" &&
          e !== null &&
          typeof (e as LocalWatchlistEntry).symbol === "string" &&
          typeof (e as LocalWatchlistEntry).companyName === "string"
      )
    : [];

  const searchHistory: string[] = Array.isArray(body?.searchHistory)
    ? body.searchHistory.filter((q: unknown): q is string => typeof q === "string")
    : [];

  if (watchlist.length > 0) watchlistItems.mergeFromLocal(userId, watchlist);
  if (searchHistory.length > 0) await searchHistoryItems.mergeFromLocal(userId, searchHistory);

  return new NextResponse(null, { status: 204 });
}
