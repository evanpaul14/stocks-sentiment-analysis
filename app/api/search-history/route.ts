import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import * as searchHistoryItems from "@/lib/db/queries/searchHistoryItems";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const entries = await searchHistoryItems.listForUser(userId);
  return NextResponse.json({ queries: entries.map((e) => e.query) });
}

export async function POST(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const body = await request.json().catch(() => null);
  const query = typeof body?.query === "string" ? body.query.trim() : "";
  if (!query) return NextResponse.json({ error: "query is required" }, { status: 400 });

  await searchHistoryItems.record(userId, query);
  return new NextResponse(null, { status: 204 });
}
