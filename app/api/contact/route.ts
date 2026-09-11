import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { isMailgunEnabled, isValidEmail, sendContactMessage } from "@/lib/integrations/mailgun";
import { verifyTurnstileToken } from "@/lib/integrations/turnstile";

const MAX_MESSAGE_LENGTH = 5000;

async function handler(request: NextRequest) {
  if (!isMailgunEnabled()) {
    return NextResponse.json({ error: "Contact form is not configured" }, { status: 503 });
  }

  const body = await request.json().catch(() => null);
  const name = typeof body?.name === "string" ? body.name.trim() : "";
  const email = typeof body?.email === "string" ? body.email.trim() : "";
  const message = typeof body?.message === "string" ? body.message.trim() : "";
  const turnstileToken = typeof body?.turnstileToken === "string" ? body.turnstileToken : "";

  if (!name || !isValidEmail(email) || !message) {
    return NextResponse.json(
      { error: "Name, a valid email, and a message are required" },
      { status: 400 }
    );
  }
  if (message.length > MAX_MESSAGE_LENGTH) {
    return NextResponse.json({ error: "Message is too long" }, { status: 400 });
  }

  const remoteIp = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  if (!(await verifyTurnstileToken(turnstileToken, remoteIp))) {
    return NextResponse.json({ error: "Verification failed — please try again" }, { status: 400 });
  }

  try {
    await sendContactMessage({ name, fromEmail: email, message });
    return NextResponse.json({ sent: true });
  } catch (error) {
    console.error("[api/contact] failed", error);
    return NextResponse.json({ error: "Failed to send message" }, { status: 502 });
  }
}

export const POST = withRateLimit({ routeName: "contact", limit: 5, windowMs: 60 * 60_000 }, handler);
