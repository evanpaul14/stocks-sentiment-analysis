import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { isAuthorizedAdminRequest } from "@/lib/auth/adminToken";
import {
  generateAndPersistMarketSummary,
  ensureMarketSummaryEmailSent,
} from "@/lib/cron/jobs/marketSummary";
import { todayInEastern } from "@/lib/utils/dates";

/** Admin-only: force-regenerate (overwrite) the market summary for a given date. */
async function handler(request: NextRequest) {
  if (!isAuthorizedAdminRequest(request)) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await request.json().catch(() => null);
  const dateKey =
    typeof body?.date === "string" && body.date ? body.date : todayInEastern();

  const record = await generateAndPersistMarketSummary(dateKey);
  await ensureMarketSummaryEmailSent(record.id, record.title, record.body, record.imageUrl);

  return NextResponse.json({ market_summary: record });
}

export const POST = withRateLimit(
  { routeName: "market-summary-generate", limit: 10, windowMs: 60 * 60_000 },
  handler
);
