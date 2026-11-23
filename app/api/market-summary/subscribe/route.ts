import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import {
  addMemberToMailgunList,
  isMailgunEnabled,
  isValidEmail,
  sendMarketSummaryToRecipient,
} from "@/lib/integrations/mailgun";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import { buildMarketSummaryEmailHtml } from "@/lib/email/marketSummaryEmail";

async function handler(request: NextRequest) {
  if (!isMailgunEnabled()) {
    return NextResponse.json({ error: "Email subscription is not configured" }, { status: 503 });
  }

  const body = await request.json().catch(() => null);
  const email = typeof body?.email === "string" ? body.email.trim() : "";

  if (!isValidEmail(email)) {
    return NextResponse.json({ error: "Invalid email address" }, { status: 400 });
  }

  try {
    await addMemberToMailgunList(email);

    const latest = await marketSummary.getLatest();
    if (latest) {
      await sendMarketSummaryToRecipient(
        email,
        latest.title,
        latest.body,
        buildMarketSummaryEmailHtml(latest)
      );
    }

    return NextResponse.json({ subscribed: true });
  } catch (error) {
    console.error("[api/market-summary/subscribe] failed", error);
    return NextResponse.json({ error: "Subscription failed" }, { status: 502 });
  }
}

export const POST = withRateLimit(
  { routeName: "market-summary-subscribe", limit: 6, windowMs: 60 * 60_000 },
  handler
);
