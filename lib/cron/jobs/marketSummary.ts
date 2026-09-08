import { isNyseTradingDay } from "@/lib/integrations/nyseCalendar";
import { getMarketIndexSnapshots } from "@/lib/integrations/yahoo/indices";
import { getMarketNewsDigest } from "@/lib/integrations/news/googleNews";
import { generateMarketSummaryText } from "@/lib/integrations/llm7/marketSummaryText";
import { getOrFetchUnsplashImage, hashCacheKey } from "@/lib/integrations/unsplash";
import { pingIndexNow } from "@/lib/integrations/indexnow";
import {
  dispatchMarketSummaryEmail,
  isMailgunEnabled,
} from "@/lib/integrations/mailgun";
import { formatDateLabel, todayInEastern } from "@/lib/utils/dates";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import * as sendLog from "@/lib/db/queries/marketWrapSendLog";

const MARKET_SUMMARY_MAX_HEADLINES = Number(
  process.env.MARKET_SUMMARY_MAX_HEADLINES ?? 8
);

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
    title: `Market Wrap: ${dateLabel}`,
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
