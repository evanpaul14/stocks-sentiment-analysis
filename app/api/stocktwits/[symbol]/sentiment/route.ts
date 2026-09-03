import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getStockTwitsSentimentCard } from "@/lib/integrations/stocktwits/sentimentCard";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  const { symbol } = await context.params;
  try {
    const card = await getStockTwitsSentimentCard(symbol.toUpperCase());
    return NextResponse.json(card);
  } catch (error) {
    console.error(`[api/stocktwits/sentiment] failed for ${symbol}`, error);
    return NextResponse.json(
      { sentiment: null, feed: [] },
      { status: 200 }
    );
  }
}

export const GET = withRateLimit(
  { routeName: "stocktwits-sentiment", limit: 20, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
