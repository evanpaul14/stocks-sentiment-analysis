import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUserId } from "@/lib/auth/currentUser";
import * as watchlistItems from "@/lib/db/queries/watchlistItems";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const entries = await watchlistItems.listForUser(userId);
  return NextResponse.json({ entries });
}

export async function POST(request: NextRequest) {
  const userId = await getCurrentUserId(request);
  if (!userId) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });

  const body = await request.json().catch(() => null);
  const symbol = typeof body?.symbol === "string" ? body.symbol.trim().toUpperCase() : "";
  const companyName = typeof body?.companyName === "string" ? body.companyName : "";
  if (!symbol || !companyName) {
    return NextResponse.json({ error: "symbol and companyName are required" }, { status: 400 });
  }

  watchlistItems.add(userId, symbol, companyName);
  return new NextResponse(null, { status: 204 });
}
