import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getPriceSnapshots } from "@/lib/integrations/yahoo/quote";

const MAX_SYMBOLS = 50;

/** Batched price refresh for a list of symbols — used by the watchlist and trending pages. */
async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const symbols: string[] = Array.isArray(body?.symbols)
    ? body.symbols.filter((s: unknown): s is string => typeof s === "string")
    : [];

  if (symbols.length === 0) {
    return NextResponse.json({ error: "Missing symbols" }, { status: 400 });
  }

  const uppercased = [...new Set(symbols.map((s: string) => s.toUpperCase()))].slice(
    0,
    MAX_SYMBOLS
  );

  try {
    const snapshots = await getPriceSnapshots(uppercased);
    return NextResponse.json(snapshots);
  } catch (error) {
    console.error("[api/trending/prices] failed", error);
    return NextResponse.json([], { status: 200 });
  }
}

export const POST = withRateLimit(
  { routeName: "trending-prices", limit: 120, windowMs: 60_000 },
  handler
);
