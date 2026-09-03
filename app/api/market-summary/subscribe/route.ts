import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import {
  addMemberToMailgunList,
  isMailgunEnabled,
  isValidEmail,
  sendMarketSummaryToRecipient,
} from "@/lib/integrations/mailgun";
import * as marketSummary from "@/lib/db/queries/marketWrap";

function buildEmailHtml(title: string, body: string): string {
  const paragraphs = body
    .split("\n")
    .filter(Boolean)
    .map((p) => `<p>${p}</p>`)
    .join("\n");
  return `<html><body><h1>${title}</h1>${paragraphs}</body></html>`;
}

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
        buildEmailHtml(latest.title, latest.body)
      );
    }

    return NextResponse.json({ subscribed: true });
  } catch (error) {
    console.error("[api/market-summary/subscribe] failed", error);
    return NextResponse.json({ error: "Subscription failed" }, { status: 502 });
  }
}

export const POST = withRateLimit(
  { routeName: "market-summary-subscribe", limit: 3, windowMs: 60 * 60_000 },
  handler
);
