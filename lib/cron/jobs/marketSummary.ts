import { isNyseTradingDay } from "@/lib/integrations/nyseCalendar";
import { getMarketIndexSnapshots, type IndexSnapshot } from "@/lib/integrations/yahoo/indices";
import { getMarketNewsDigest } from "@/lib/integrations/news/googleNews";
import { generateMarketSummaryText } from "@/lib/integrations/llm7/marketSummaryText";
import { getOrFetchUnsplashImage, hashCacheKey } from "@/lib/integrations/unsplash";
import { pingIndexNow } from "@/lib/integrations/indexnow";
import {
  dispatchMarketSummaryEmail,
  isMailgunEnabled,
} from "@/lib/integrations/mailgun";
import { formatDateLabel, formatDateLabelWithWeekday, todayInEastern } from "@/lib/utils/dates";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import * as sendLog from "@/lib/db/queries/marketWrapSendLog";

const MARKET_SUMMARY_MAX_HEADLINES = Number(
  process.env.MARKET_SUMMARY_MAX_HEADLINES ?? 8
);

/** Short index names for the title, e.g. "Nasdaq" instead of "Nasdaq Composite". */
const INDEX_SHORT_NAMES: Record<string, string> = {
  "^GSPC": "S&P 500",
  "^IXIC": "Nasdaq",
  "^DJI": "Dow Jones",
};

function joinWithAmpersand(items: string[]): string {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} & ${items[items.length - 1]}`;
}

/**
 * Builds a title like "Market Wrap: Friday, July 17, 2026 — Nasdaq, S&P 500 & Dow Jones".
 * Front-loads the date (the part most likely to survive Google's ~60-char title truncation
 * and the part that matches long-tail date searches) while keeping the original "Market Wrap"
 * framing and naming the indexes, which the old bare "Market Wrap: <date>" title didn't do.
 */
function buildMarketSummaryTitle(dateKey: string, indexes: IndexSnapshot[]): string {
  const weekdayDateLabel = formatDateLabelWithWeekday(dateKey);
  const names = indexes.map((index) => INDEX_SHORT_NAMES[index.symbol] ?? index.name);
  const namesSuffix = names.length > 0 ? ` — ${joinWithAmpersand(names)}` : "";
  return `Market Wrap: ${weekdayDateLabel}${namesSuffix}`;
}

/** Generates and stores a market summary for dateKey, overwriting any existing one. */
export async function generateAndPersistMarketSummary(dateKey: string) {
  const [indexes, headlines] = await Promise.all([
    getMarketIndexSnapshots(),
    getMarketNewsDigest(MARKET_SUMMARY_MAX_HEADLINES),
  ]);

  const dateLabel = formatDateLabel(dateKey);
  const body = await generateMarketSummaryText(dateLabel, indexes, headlines);
  const image = await getOrFetchUnsplashImage(
    hashCacheKey(`market-summary:${dateKey}`),
    `stock market wrap ${dateLabel}`
  );

  const record = marketSummary.upsertByDate({
    date: dateKey,
    slug: dateKey,
    title: buildMarketSummaryTitle(dateKey, indexes),
    body,
    indexSnapshotJson: JSON.stringify(indexes),
    headlinesJson: JSON.stringify(headlines),
    imageUrl: image?.imageUrl ?? null,
    imageThumbnailUrl: image?.thumbnailUrl ?? null,
    imagePhotographerName: image?.photographerName ?? null,
    imagePhotographerProfileUrl: image?.photographerProfileUrl ?? null,
  });

  const baseUrl = process.env.SITE_BASE_URL ?? "";
  await pingIndexNow([
    `${baseUrl}/market-summary/${record.slug}`,
    `${baseUrl}/market-summary/stock-market-today`,
  ]);

  return record;
}

/** Generates (or reuses) a market summary for dateKey. Idempotent per date. */
export async function ensureMarketSummaryForDate(dateKey: string) {
  const existing = await marketSummary.getByDate(dateKey);
  if (existing) return existing;
  return generateAndPersistMarketSummary(dateKey);
}

function buildEmailHtml(title: string, body: string, imageUrl?: string | null): string {
  const paragraphs = body
    .split("\n")
    .filter(Boolean)
    .map((p) => `<p>${p}</p>`)
    .join("\n");
  const imageHtml = imageUrl
    ? `<p><img src="${imageUrl}" alt="${title}" style="max-width:320px;width:100%;border-radius:12px;display:block;" /></p>`
    : "";
  return `<html><body><h1>${title}</h1>${imageHtml}${paragraphs}<p><a href="%unsubscribe_url%">Unsubscribe</a></p></body></html>`;
}

export async function ensureMarketSummaryEmailSent(
  summaryId: number,
  title: string,
  body: string,
  imageUrl?: string | null
) {
  if (!isMailgunEnabled()) return;
  if (!sendLog.tryClaimSend(summaryId)) return;

  try {
    await dispatchMarketSummaryEmail(title, body, buildEmailHtml(title, body, imageUrl));
    sendLog.recordSend(summaryId, "sent");
  } catch (error) {
    const message = error instanceof Error ? error.message.slice(0, 250) : "unknown error";
    sendLog.recordSend(summaryId, "failed", message);
    console.error("[market-summary] email send failed", error);
  }
}

/** The daily cron job: generate today's summary (if a trading day) and email it. */
export async function runMarketSummaryJob() {
  const today = todayInEastern();
  if (!isNyseTradingDay(today)) {
    console.log(`[market-summary] ${today} is not a trading day, skipping`);
    return;
  }

  const record = await ensureMarketSummaryForDate(today);
  await ensureMarketSummaryEmailSent(record.id, record.title, record.body, record.imageUrl);
}
