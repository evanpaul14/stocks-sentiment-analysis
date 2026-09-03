import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getSentimentPriceOverlay } from "@/lib/sentiment/sentimentPriceOverlay";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await context.params;
  try {
    const overlay = await getSentimentPriceOverlay(symbol);
    return NextResponse.json(overlay);
  } catch (error) {
    console.error(`[api/sentiment-history] failed for ${symbol}`, error);
    return NextResponse.json([], { status: 200 });
  }
}

export const GET = withRateLimit(
  { routeName: "sentiment-history", limit: 20, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
