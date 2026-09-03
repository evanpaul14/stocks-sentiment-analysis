import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { getMarketIndexSnapshots } from "@/lib/integrations/yahoo/indices";
import { TtlCache } from "@/lib/cache/memory";

const CACHE_TTL_MS = Number(process.env.MARKET_WEEK_GLANCE_TTL_SECONDS ?? 300) * 1000;
const cache = new TtlCache<Awaited<ReturnType<typeof getMarketIndexSnapshots>>>(CACHE_TTL_MS);

async function handler() {
  const snapshots = await cache.getOrCompute("week-glance", getMarketIndexSnapshots);
  return NextResponse.json(snapshots);
}

export const GET = withRateLimit(
  { routeName: "market-summary-week-glance", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
