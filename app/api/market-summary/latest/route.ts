import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import * as marketSummary from "@/lib/db/queries/marketWrap";

async function handler() {
  const latest = await marketSummary.getLatest();
  if (!latest) {
    return NextResponse.json({ error: "No market summary available yet" }, { status: 404 });
  }
  return NextResponse.json(latest);
}

export const GET = withRateLimit(
  { routeName: "market-summary-latest", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
