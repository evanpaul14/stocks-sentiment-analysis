import { NextResponse, type NextRequest } from "next/server";
import { getCurrentUser } from "@/lib/auth/currentUser";
import {
  getMailgunListMember,
  addMemberToMailgunList,
  isMailgunEnabled,
} from "@/lib/integrations/mailgun";

export const dynamic = "force-dynamic";

/** Reads the signed-in user's current market-summary subscription status. */
export async function GET(request: NextRequest) {
  const user = await getCurrentUser(request);
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  if (!user.email) return NextResponse.json({ error: "no email on account" }, { status: 400 });

  if (!isMailgunEnabled()) {
    return NextResponse.json({ error: "Email subscription is not configured" }, { status: 503 });
  }

  try {
    const member = await getMailgunListMember(user.email);
    return NextResponse.json({ subscribed: member?.subscribed ?? false });
  } catch (error) {
    console.error("[api/account/market-summary-subscription] lookup failed", error);
    return NextResponse.json({ error: "Failed to look up subscription" }, { status: 502 });
  }
}

/** Subscribes/unsubscribes the signed-in user's account email to the market-summary list. */
export async function PUT(request: NextRequest) {
  const user = await getCurrentUser(request);
  if (!user) return NextResponse.json({ error: "unauthenticated" }, { status: 401 });
  if (!user.email) return NextResponse.json({ error: "no email on account" }, { status: 400 });

  if (!isMailgunEnabled()) {
    return NextResponse.json({ error: "Email subscription is not configured" }, { status: 503 });
  }

  const body = await request.json().catch(() => null);
  if (typeof body?.subscribed !== "boolean") {
    return NextResponse.json({ error: "subscribed (boolean) is required" }, { status: 400 });
  }

  try {
    await addMemberToMailgunList(user.email, body.subscribed);
    return NextResponse.json({ subscribed: body.subscribed });
  } catch (error) {
    console.error("[api/account/market-summary-subscription] update failed", error);
    return NextResponse.json({ error: "Failed to update subscription" }, { status: 502 });
  }
}
