import { stocktwitsFetch } from "./fetchClient";

export interface StockTwitsMessage {
  id: number;
  body: string;
  created_at: string;
  user: { username: string; avatar_url: string };
  entities?: {
    sentiment?: { basic?: string };
    media?: Array<Record<string, unknown>>;
    links?: Array<Record<string, unknown>>;
  };
  reshare_message?: {
    reshared_deleted?: boolean;
    reshared_user_deleted?: boolean;
    message?: StockTwitsMessage;
  };
}

interface StreamPage {
  messages: StockTwitsMessage[];
  cursor: { more: boolean; max: number };
}

function streamUrl(symbol: string): string {
  return `https://api.stocktwits.com/api/2/streams/symbol/${encodeURIComponent(symbol)}.json`;
}

async function fetchPage(
  symbol: string,
  filter: string,
  limit: number,
  maxId?: number
): Promise<StreamPage> {
  const data = (await stocktwitsFetch(streamUrl(symbol), {
    filter,
    limit,
    max: maxId,
  })) as StreamPage;
  return data;
}

export interface CollectMessagesOptions {
  filter?: string;
  batchLimit?: number;
  maxBatches?: number;
  targetSentimentMessages?: number;
}

/**
 * Pages backward through a symbol's message stream (via the "max" cursor)
 * until enough sentiment-tagged messages are collected or the batch/page
 * limits are hit. Dedupes by message id.
 */
export async function collectStockTwitsMessages(
  symbol: string,
  options: CollectMessagesOptions = {}
): Promise<StockTwitsMessage[]> {
  const filter = options.filter ?? "top";
  const batchLimit = Math.min(options.batchLimit ?? 100, 100);
  const maxBatches = Math.min(options.maxBatches ?? 12, 20);
  const target = Math.min(options.targetSentimentMessages ?? 50, 250);

  const seen = new Set<number>();
  const collected: StockTwitsMessage[] = [];
  let cursor: number | undefined;
  let sentimentTaggedCount = 0;

  for (let batch = 0; batch < maxBatches; batch++) {
    const page = await fetchPage(symbol, filter, batchLimit, cursor);
    if (page.messages.length === 0) break;

    for (const message of page.messages) {
      if (seen.has(message.id)) continue;
      seen.add(message.id);
      collected.push(message);
      if (message.entities?.sentiment?.basic) sentimentTaggedCount++;
    }

    if (sentimentTaggedCount >= target) break;

    const oldestId = Math.min(...page.messages.map((m) => m.id));
    if (cursor !== undefined && oldestId >= cursor) break; // no progress, stop
    cursor = oldestId;
  }

  return collected;
}
