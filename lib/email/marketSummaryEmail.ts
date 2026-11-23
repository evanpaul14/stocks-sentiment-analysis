import { escapeHtml } from "@/lib/utils/html";
import type { IndexSnapshot } from "@/lib/integrations/yahoo/indices";

export interface MarketSummaryEmailInput {
  title: string;
  body: string;
  slug: string;
  indexSnapshotJson: string;
  imageUrl?: string | null;
  imagePhotographerName?: string | null;
  imagePhotographerProfileUrl?: string | null;
}

const BG = "#f4f1ea";
const CARD = "#ffffff";
const FG = "#1c1915";
const BORDER = "#e6e0d4";
const BULLISH = "#2f7d5c";
const BEARISH = "#b4533c";
const MUTED = "#6f675c";
const GOLD = "#ddb049";
const SERIF = "Fraunces,Georgia,'Times New Roman',serif";
const SANS = "'Hanken Grotesk',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif";

function parseIndexes(json: string): IndexSnapshot[] {
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function formatPercent(value: number | null): { text: string; color: string } {
  if (value == null) return { text: "n/a", color: MUTED };
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return {
    text: `${sign}${Math.abs(value).toFixed(2)}%`,
    color: value > 0 ? BULLISH : value < 0 ? BEARISH : MUTED,
  };
}

/** One compact row: each index name over its percent change today. */
function indexStrip(indexes: IndexSnapshot[]): string {
  if (indexes.length === 0) return "";
  const cells = indexes
    .map((index, i) => {
      const day = formatPercent(index.changePercent);
      const border = i > 0 ? `border-left:1px solid ${BORDER};` : "";
      return `<td align="center" style="padding:10px 4px;${border}">
<div style="font-size:12px;color:${MUTED};">${escapeHtml(index.name.replace(/ (Composite|Industrial Average)$/, ""))}</div>
<div style="font-size:18px;font-weight:700;color:${day.color};padding-top:2px;">${day.text}</div>
</td>`;
    })
    .join("\n");
  return `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 24px;border-top:1px solid ${BORDER};border-bottom:1px solid ${BORDER};"><tr>${cells}</tr></table>`;
}

/** "Market Wrap: <date> — <indexes>" -> "Market Wrap: <date>". */
function headingFromTitle(title: string): string {
  return title.split(" — ")[0];
}

export function buildMarketSummaryEmailHtml(input: MarketSummaryEmailInput): string {
  const baseUrl = process.env.SITE_BASE_URL ?? "";
  const pageUrl = `${baseUrl}/market-summary/${encodeURIComponent(input.slug)}`;
  const heading = headingFromTitle(input.title);
  const indexes = parseIndexes(input.indexSnapshotJson);

  const paragraphs = input.body
    .split("\n")
    .map((p) => p.trim())
    .filter(Boolean)
    .map(
      (p) =>
        `<p style="margin:0 0 18px;font-size:16px;line-height:1.65;color:${FG};">${escapeHtml(p)}</p>`
    )
    .join("\n");

  const credit = input.imagePhotographerName
    ? `<p style="margin:6px 0 0;font-size:11px;color:${MUTED};">Photo by ${
        input.imagePhotographerProfileUrl
          ? `<a href="${escapeHtml(input.imagePhotographerProfileUrl)}" style="color:${MUTED};">${escapeHtml(input.imagePhotographerName)}</a>`
          : escapeHtml(input.imagePhotographerName)
      } on Unsplash</p>`
    : "";
  const image = input.imageUrl
    ? `<img src="${escapeHtml(input.imageUrl)}" alt="" width="240" style="display:block;width:100%;max-width:240px;height:auto;border:0;" />${credit}<div style="height:24px;line-height:24px;">&nbsp;</div>`
    : "";

  // Mailgun substitutes %unsubscribe_url% at send time.
  const unsubscribe = ` &nbsp;·&nbsp; <a href="%unsubscribe_url%" style="color:${MUTED};text-decoration:underline;">Unsubscribe</a>`;

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>${escapeHtml(input.title)}</title></head>
<body style="margin:0;padding:0;background:${BG};-webkit-text-size-adjust:100%;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" bgcolor="${BG}" style="background:${BG};">
<tr><td align="center" style="padding:24px 12px;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:600px;font-family:${SANS};">
<tr><td bgcolor="${CARD}" style="background:${CARD};border:1px solid ${BORDER};padding:32px 32px 8px;">
<h1 style="margin:0 0 24px;font-family:${SERIF};font-size:28px;line-height:1.15;font-weight:500;letter-spacing:-0.02em;color:${FG};">${escapeHtml(heading)}</h1>
${image}
${indexStrip(indexes)}
${paragraphs}
<p style="margin:8px 0 28px;"><a href="${escapeHtml(pageUrl)}" style="display:inline-block;background:${GOLD};color:#161107;text-decoration:none;font-size:14px;font-weight:600;padding:11px 20px;">Read on Stock Sentiment App</a></p>
</td></tr>
<tr><td align="center" style="padding:20px 8px 0;font-size:12px;line-height:1.6;color:${MUTED};">
Daily recap from Stock Sentiment. For information only, not investment advice.${unsubscribe}
</td></tr>
</table>
</td></tr>
</table>
</body></html>`;
}
