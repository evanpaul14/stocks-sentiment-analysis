import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getHistoricalPrices } from "@/lib/integrations/yahoo/historical";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ symbol: string; period: string }> }
) {
  const { symbol, period } = await context.params;
  try {
    const prices = await getHistoricalPrices(symbol.toUpperCase(), period);
    return NextResponse.json(prices);
  } catch (error) {
    console.error(`[api/historical] failed for ${symbol}/${period}`, error);
    return NextResponse.json({ error: "Historical data lookup failed" }, { status: 502 });
  }
}

export const GET = withRateLimit(
  { routeName: "historical", limit: 50, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
