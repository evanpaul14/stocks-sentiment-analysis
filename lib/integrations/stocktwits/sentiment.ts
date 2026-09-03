import type { StockTwitsMessage } from "./stream";

export interface StockTwitsSentimentResult {
  overallSentiment: "bullish" | "bearish" | "neutral";
  bullishCount: number;
  bearishCount: number;
  bullishPercent: number;
  bearishPercent: number;
  taggedMessageCount: number;
  totalMessageCount: number;
}

/**
 * StockTwits only tags "bullish"/"bearish" (no neutral from the API itself).
 * Majority count wins; ties or zero tagged messages default to neutral.
 */
export function calculateStockTwitsSentiment(
  messages: StockTwitsMessage[]
): StockTwitsSentimentResult {
  let bullishCount = 0;
  let bearishCount = 0;

  for (const message of messages) {
    const basic = message.entities?.sentiment?.basic?.toLowerCase();
    if (basic === "bullish") bullishCount++;
    else if (basic === "bearish") bearishCount++;
  }

  const taggedMessageCount = bullishCount + bearishCount;
  const overallSentiment =
    taggedMessageCount === 0
      ? "neutral"
      : bullishCount > bearishCount
        ? "bullish"
        : bearishCount > bullishCount
          ? "bearish"
          : "neutral";

  return {
    overallSentiment,
    bullishCount,
    bearishCount,
    bullishPercent:
      taggedMessageCount === 0
        ? 0
        : Number(((bullishCount / taggedMessageCount) * 100).toFixed(2)),
    bearishPercent:
      taggedMessageCount === 0
        ? 0
        : Number(((bearishCount / taggedMessageCount) * 100).toFixed(2)),
    taggedMessageCount,
    totalMessageCount: messages.length,
  };
}
