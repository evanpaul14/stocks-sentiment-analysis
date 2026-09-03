import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getPriceSnapshot } from "@/lib/integrations/yahoo/quote";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await context.params;
  try {
    const snapshot = await getPriceSnapshot(symbol.toUpperCase());
    return NextResponse.json(snapshot);
  } catch (error) {
    console.error(`[api/quote] failed for ${symbol}`, error);
    return NextResponse.json({ error: "Quote lookup failed" }, { status: 502 });
  }
}

export const GET = withRateLimit(
  { routeName: "quote", limit: 300, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
