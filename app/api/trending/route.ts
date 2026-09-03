import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getAllTrendingSourceData } from "@/lib/trending/getTrendingSourceData";

async function handler() {
  const data = await getAllTrendingSourceData();
  return NextResponse.json(data);
}

export const GET = withRateLimit(
  { routeName: "trending-all", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
