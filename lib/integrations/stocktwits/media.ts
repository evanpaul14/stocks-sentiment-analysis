import type { StockTwitsMessage } from "./stream";

export interface FeedMessage {
  id: number;
  createdAt: string;
  username: string;
  avatarUrl: string;
  sentiment: "bullish" | "bearish" | null;
  body: string;
  imageUrls: string[];
  messageUrl: string;
  profileUrl: string;
  quotedPost: FeedMessage | null;
}

const IMAGE_KEY_PREFERENCE = ["original", "large", "url", "medium", "thumb"];

function extractImageUrls(message: StockTwitsMessage): string[] {
  const urls: string[] = [];

  for (const media of message.entities?.media ?? []) {
    if (media.provider === "giphy" && media.provider_id) {
      urls.push(`https://media.giphy.com/media/${media.provider_id}/giphy.gif`);
      continue;
    }
    for (const key of IMAGE_KEY_PREFERENCE) {
      if (typeof media[key] === "string") {
        urls.push(media[key] as string);
        break;
      }
    }
  }

  for (const link of message.entities?.links ?? []) {
    const candidate = link.image ?? link.image_url ?? link.images;
    if (typeof candidate === "string") urls.push(candidate);
    else if (Array.isArray(candidate) && typeof candidate[0] === "string") {
      urls.push(candidate[0]);
    }
  }

  return urls;
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  quot: '"',
  apos: "'",
  lt: "<",
  gt: ">",
  nbsp: " ",
};

/**
 * StockTwits bodies arrive HTML-escaped (`don&#39;t`, `&quot;`, `&amp;` in
 * URLs) but React renders text literally, so decode them. Single pass, so
 * `&amp;quot;` correctly becomes `&quot;` rather than `"`.
 */
function decodeHtmlEntities(text: string): string {
  return text.replace(/&(#\d+|#x[0-9a-f]+|[a-z]+);/gi, (match, entity: string) => {
    if (entity[0] === "#") {
      const code =
        entity[1] === "x" || entity[1] === "X"
          ? parseInt(entity.slice(2), 16)
          : parseInt(entity.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff
        ? String.fromCodePoint(code)
        : match;
    }
    return NAMED_ENTITIES[entity.toLowerCase()] ?? match;
  });
}

/**
 * Decodes entities and tidies whitespace: trims each line, drops lone "."
 * spacer lines users add, and collapses runs of blank lines to one.
 */
export function cleanStockTwitsBody(body: string): string {
  return decodeHtmlEntities(body)
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== ".")
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function toFeedMessage(message: StockTwitsMessage): FeedMessage {
  const rawSentiment = message.entities?.sentiment?.basic?.toLowerCase();
  const sentiment =
    rawSentiment === "bullish" || rawSentiment === "bearish" ? rawSentiment : null;

  const reshare = message.reshare_message;
  const quotedPost =
    reshare &&
    !reshare.reshared_deleted &&
    !reshare.reshared_user_deleted &&
    reshare.message
      ? toFeedMessage(reshare.message)
      : null;

  return {
    id: message.id,
    createdAt: message.created_at,
    username: message.user.username,
    avatarUrl: message.user.avatar_url,
    sentiment,
    body: cleanStockTwitsBody(message.body ?? ""),
    imageUrls: extractImageUrls(message),
    messageUrl: `https://stocktwits.com/symbol/message/${message.id}`,
    profileUrl: `https://stocktwits.com/${message.user.username}`,
    quotedPost,
  };
}

/** Builds the scrollable message feed shown on the StockTwits sentiment card. */
export function buildStockTwitsMessageFeed(
  messages: StockTwitsMessage[],
  feedLimit = 40
): FeedMessage[] {
  return messages.slice(0, Math.min(feedLimit, 100)).map(toFeedMessage);
}
