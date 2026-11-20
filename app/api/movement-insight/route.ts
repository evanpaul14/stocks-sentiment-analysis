import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { buildMovementInsight } from "@/lib/integrations/llm7/movementInsight";
import { getStockInfo } from "@/lib/integrations/yahoo/quote";
import { isPlausibleSymbol } from "@/lib/utils/tickers";

/**
 * Only the symbol is taken from the client. The insight is cached per symbol
 * and server-rendered for every visitor of that stock page, so company name
 * and % change are looked up here rather than trusted from the request body —
 * otherwise any caller could seed the shared cache with their own text.
 */
async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const symbol = typeof body?.symbol === "string" ? body.symbol.trim().toUpperCase() : "";

  if (!isPlausibleSymbol(symbol)) {
    return NextResponse.json({ error: "Missing or invalid symbol" }, { status: 400 });
  }

  try {
    const stockInfo = await getStockInfo(symbol);
    if (stockInfo.regularMarketChangePercent == null) {
      return NextResponse.json({ movement_insight: null });
    }
    const insight = await buildMovementInsight(
      stockInfo.symbol,
      stockInfo.companyName,
      stockInfo.regularMarketChangePercent
    );
    return NextResponse.json({ movement_insight: insight });
  } catch (error) {
    console.error("[api/movement-insight] failed", error);
    return NextResponse.json({ movement_insight: null }, { status: 200 });
  }
}

export const POST = withRateLimit(
  { routeName: "movement-insight", limit: 20, windowMs: 60_000 },
  handler
);
