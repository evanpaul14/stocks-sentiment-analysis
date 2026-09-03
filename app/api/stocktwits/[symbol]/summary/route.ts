import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getStockTwitsSummary } from "@/lib/integrations/stocktwits/summary";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await context.params;
  try {
    const summary = await getStockTwitsSummary(symbol.toUpperCase());
    return NextResponse.json({ summary });
  } catch (error) {
    console.error(`[api/stocktwits/summary] failed for ${symbol}`, error);
    return NextResponse.json({ summary: null }, { status: 200 });
  }
}

export const GET = withRateLimit(
  { routeName: "stocktwits-summary", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
