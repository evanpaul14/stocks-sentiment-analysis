import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getTrendingSourceData, type TrendingSource } from "@/lib/trending/getTrendingSourceData";

const VALID_SOURCES: TrendingSource[] = ["stocktwits", "reddit", "volume"];

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ source: string }> }
) {
  const { source } = await context.params;
  if (!VALID_SOURCES.includes(source as TrendingSource)) {
    return NextResponse.json({ error: "Unknown source" }, { status: 404 });
  }
  const data = await getTrendingSourceData(source as TrendingSource);
  return NextResponse.json(data);
}

export const GET = withRateLimit(
  { routeName: "trending-source", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
