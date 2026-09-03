import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import * as marketSummary from "@/lib/db/queries/marketWrap";

async function handler() {
  const archive = await marketSummary.listArchive(60);
  return NextResponse.json(archive);
}

export const GET = withRateLimit(
  { routeName: "market-summary-archive", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
