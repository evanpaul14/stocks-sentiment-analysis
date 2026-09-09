const API_BASE = "https://api.mailgun.net/v3";

function isEnabled(): boolean {
  return Boolean(process.env.MAILGUN_API_KEY && process.env.MAILGUN_DOMAIN);
}

function authHeader(): string {
  return `Basic ${Buffer.from(`api:${process.env.MAILGUN_API_KEY}`).toString("base64")}`;
}

export class MailgunNotConfiguredError extends Error {
  constructor() {
    super("Mailgun is not configured");
    this.name = "MailgunNotConfiguredError";
  }
}

const EMAIL_REGEX = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;

export function isValidEmail(email: string): boolean {
  return EMAIL_REGEX.test(email);
}

/** Adds (or upserts) a subscriber to the market-summary mailing list. */
export async function addMemberToMailgunList(email: string): Promise<void> {
  if (!isEnabled()) throw new MailgunNotConfiguredError();

  const listAddress =
    process.env.MAILGUN_MARKET_LIST_ADDRESS ??
    `marketsummary@${process.env.MAILGUN_DOMAIN}`;

  const response = await fetch(
    `${API_BASE}/lists/${encodeURIComponent(listAddress)}/members`,
    {
      method: "POST",
      headers: {
        Authorization: authHeader(),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ address: email, subscribed: "yes", upsert: "yes" }),
      signal: AbortSignal.timeout(10_000),
    }
  );

  if (!response.ok) {
    throw new Error(`Mailgun list subscribe failed: ${response.status}`);
  }
}

/**
 * Removes an address from this domain's unsubscribe suppression list.
 * Mailgun enforces that suppression at the SMTP layer regardless of any
 * per-message override header, so account emails (verification, password
 * reset) that must reach someone even after they unsubscribed from the
 * market-wrap newsletter need this instead. This only lifts Mailgun's
 * domain-wide delivery block — it does not touch mailing-list membership
 * (addMemberToMailgunList), so it can't accidentally re-subscribe anyone
 * to the newsletter.
 */
async function clearUnsubscribeSuppression(email: string): Promise<void> {
  await fetch(
    `${API_BASE}/${process.env.MAILGUN_DOMAIN}/unsubscribes/${encodeURIComponent(email)}`,
    {
      method: "DELETE",
      headers: { Authorization: authHeader() },
      signal: AbortSignal.timeout(10_000),
    }
  ).catch(() => {
    // Best-effort — if this fails (e.g. address wasn't suppressed, a 404),
    // the send below still gets attempted normally.
  });
}

interface SendEmailOptions {
  to: string;
  subject: string;
  text: string;
  html: string;
  /** See clearUnsubscribeSuppression — only for account emails. */
  bypassUnsubscribeSuppression?: boolean;
}

async function sendEmail({
  to,
  subject,
  text,
  html,
  bypassUnsubscribeSuppression,
}: SendEmailOptions): Promise<void> {
  if (!isEnabled()) throw new MailgunNotConfiguredError();

  if (bypassUnsubscribeSuppression) {
    await clearUnsubscribeSuppression(to);
  }

  const fromAddress =
    process.env.MAILGUN_FROM_EMAIL ??
    `Stock Sentiment App <postmaster@${process.env.MAILGUN_DOMAIN}>`;

  const response = await fetch(
    `${API_BASE}/${process.env.MAILGUN_DOMAIN}/messages`,
    {
      method: "POST",
      headers: {
        Authorization: authHeader(),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({ from: fromAddress, to, subject, text, html }),
      signal: AbortSignal.timeout(10_000),
    }
  );

  if (!response.ok) {
    throw new Error(`Mailgun send failed: ${response.status}`);
  }
}

const listAddress = () =>
  process.env.MAILGUN_MARKET_LIST_ADDRESS ??
  `marketsummary@${process.env.MAILGUN_DOMAIN}`;

/** Broadcasts the daily market summary to the whole mailing list. */
export async function dispatchMarketSummaryEmail(
  subject: string,
  text: string,
  html: string
): Promise<void> {
  await sendEmail({
    to: `Market Summary Subscribers <${listAddress()}>`,
    subject,
    text,
    html,
  });
}

/** Sends the latest summary directly to one new subscriber, on signup. */
export async function sendMarketSummaryToRecipient(
  recipient: string,
  subject: string,
  text: string,
  html: string
): Promise<void> {
  await sendEmail({ to: recipient, subject, text, html });
}

/**
 * Transactional account emails — sent to a single recipient via the plain
 * /messages endpoint, never via addMemberToMailgunList, so verifying an
 * address or resetting a password never enrolls anyone in the newsletter.
 */
export async function sendVerificationEmail(
  to: string,
  verifyUrl: string
): Promise<void> {
  await sendEmail({
    to,
    subject: "Verify your email",
    text: `Confirm your email address to finish setting up your account: ${verifyUrl}\n\nIf you didn't request this, you can ignore this email.`,
    html: `<p>Confirm your email address to finish setting up your account:</p><p><a href="${verifyUrl}">${verifyUrl}</a></p><p>If you didn't request this, you can ignore this email.</p>`,
    bypassUnsubscribeSuppression: true,
  });
}

export async function sendPasswordResetEmail(
  to: string,
  resetUrl: string
): Promise<void> {
  await sendEmail({
    to,
    subject: "Reset your password",
    text: `Reset your password: ${resetUrl}\n\nThis link expires in 1 hour. If you didn't request this, you can ignore this email.`,
    html: `<p>Reset your password:</p><p><a href="${resetUrl}">${resetUrl}</a></p><p>This link expires in 1 hour. If you didn't request this, you can ignore this email.</p>`,
    bypassUnsubscribeSuppression: true,
  });
}

export { isEnabled as isMailgunEnabled };
