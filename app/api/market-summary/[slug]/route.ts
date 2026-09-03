import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import * as marketSummary from "@/lib/db/queries/marketWrap";

async function handler(
  _request: NextRequest,
  context: { params: Promise<{ slug: string }> }
) {
  const { slug } = await context.params;
  const record = await marketSummary.getBySlug(slug);
  if (!record) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  return NextResponse.json(record);
}

export const GET = withRateLimit(
  { routeName: "market-summary-slug", limit: 30, windowMs: 60_000 },
  handler as (request: NextRequest, context: unknown) => Promise<Response>
);
