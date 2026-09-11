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

const listAddress = () =>
  process.env.MAILGUN_MARKET_LIST_ADDRESS ??
  `marketsummary@${process.env.MAILGUN_DOMAIN}`;

/** Adds (or upserts) a subscriber to the market-summary mailing list. */
export async function addMemberToMailgunList(
  email: string,
  subscribed: boolean = true
): Promise<void> {
  if (!isEnabled()) throw new MailgunNotConfiguredError();

  const response = await fetch(
    `${API_BASE}/lists/${encodeURIComponent(listAddress())}/members`,
    {
      method: "POST",
      headers: {
        Authorization: authHeader(),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        address: email,
        subscribed: subscribed ? "yes" : "no",
        upsert: "yes",
      }),
      signal: AbortSignal.timeout(10_000),
    }
  );

  if (!response.ok) {
    throw new Error(`Mailgun list subscribe failed: ${response.status}`);
  }
}

/** Looks up a subscriber's current membership on the market-summary list, if any. */
export async function getMailgunListMember(
  email: string
): Promise<{ subscribed: boolean } | null> {
  if (!isEnabled()) throw new MailgunNotConfiguredError();

  const response = await fetch(
    `${API_BASE}/lists/${encodeURIComponent(listAddress())}/members/${encodeURIComponent(email)}`,
    {
      headers: { Authorization: authHeader() },
      signal: AbortSignal.timeout(10_000),
    }
  );

  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Mailgun member lookup failed: ${response.status}`);
  }

  const data = await response.json();
  return { subscribed: Boolean(data.member?.subscribed) };
}

interface SendEmailOptions {
  to: string;
  subject: string;
  text: string;
  html: string;
  /** Bypass the account-wide unsubscribe suppression list — for transactional mail that isn't part of a mailing list. */
  skipUnsubscribe?: boolean;
}

async function sendEmail({ to, subject, text, html, skipUnsubscribe }: SendEmailOptions): Promise<void> {
  if (!isEnabled()) throw new MailgunNotConfiguredError();

  const fromAddress =
    process.env.MAILGUN_FROM_EMAIL ??
    `Stock Sentiment App <postmaster@${process.env.MAILGUN_DOMAIN}>`;

  const params = new URLSearchParams({ from: fromAddress, to, subject, text, html });
  if (skipUnsubscribe) params.set("o:skip-unsubscribe", "true");

  const response = await fetch(
    `${API_BASE}/${process.env.MAILGUN_DOMAIN}/messages`,
    {
      method: "POST",
      headers: {
        Authorization: authHeader(),
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: params,
      signal: AbortSignal.timeout(10_000),
    }
  );

  if (!response.ok) {
    throw new Error(`Mailgun send failed: ${response.status}`);
  }
}

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

/** Forwards a /contact form submission to the site operator's inbox. */
export async function sendContactMessage(options: {
  name: string;
  fromEmail: string;
  message: string;
}): Promise<void> {
  const recipient = process.env.CONTACT_RECIPIENT_EMAIL;
  if (!recipient) throw new MailgunNotConfiguredError();

  const text = `From: ${options.name} <${options.fromEmail}>\n\n${options.message}`;
  const html = `<p><strong>From:</strong> ${options.name} &lt;${options.fromEmail}&gt;</p><p>${options.message.replace(/\n/g, "<br>")}</p>`;

  await sendEmail({
    to: recipient,
    subject: `Contact form: ${options.name}`,
    text,
    html,
    skipUnsubscribe: true,
  });
}

export { isEnabled as isMailgunEnabled };
