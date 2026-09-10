import { llm7Client, llm7Model } from "./client";
import { getLimiter } from "@/lib/ratelimit/tokenBucket";
import type { SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";

// Matches v1's Flask rate limit on the sentiment SEO page route
// (`@limiter.limit("20 per minute")`), which gated the same llm7 call.
const seoGenerationLimiter = getLimiter("llm7-seo-sentiment", 20, 60_000);

export interface SeoPageSections {
  intro: string;
  sentimentSummary: string;
  prediction: string;
}

/** Concrete, ticker-specific facts pulled straight from the overlay data — no LLM
 * required to compute these, so both the prompt and the no-LLM fallback can cite
 * real numbers instead of generic boilerplate. */
interface SeoDataPoints {
  totalArticles: number;
  daysOfData: number;
  recentAvgSentiment: number;
  positiveDayCount: number;
  negativeDayCount: number;
  neutralDayCount: number;
  priceChangePercent: number | null;
  priceRangeLow: number | null;
  priceRangeHigh: number | null;
}

function buildDataPoints(overlay: SentimentPricePoint[]): SeoDataPoints {
  const totalArticles = overlay.reduce((sum, p) => sum + p.articleCount, 0);
  const withSentiment = overlay.filter((p) => p.averageSentiment != null);
  const recent14 = withSentiment.slice(-14);
  const recentAvgSentiment = recent14.length
    ? recent14.reduce((sum, p) => sum + (p.averageSentiment ?? 0), 0) / recent14.length
    : 0;

  let positiveDayCount = 0;
  let negativeDayCount = 0;
  let neutralDayCount = 0;
  for (const p of withSentiment) {
    const score = p.averageSentiment ?? 0;
    if (score > 0.2) positiveDayCount++;
    else if (score < -0.2) negativeDayCount++;
    else neutralDayCount++;
  }

  const prices = overlay.map((p) => p.price).filter((p): p is number => p != null);
  const priceRangeLow = prices.length ? Math.min(...prices) : null;
  const priceRangeHigh = prices.length ? Math.max(...prices) : null;
  const priceChangePercent =
    prices.length >= 2 ? ((prices[prices.length - 1] - prices[0]) / prices[0]) * 100 : null;

  return {
    totalArticles,
    daysOfData: overlay.length,
    recentAvgSentiment,
    positiveDayCount,
    negativeDayCount,
    neutralDayCount,
    priceChangePercent,
    priceRangeLow,
    priceRangeHigh,
  };
}

function formatDataPointsForPrompt(dp: SeoDataPoints): string {
  const lines = [
    `${dp.totalArticles} news articles analyzed over the past ${dp.daysOfData} days`,
    `Sentiment breakdown across those days: ${dp.positiveDayCount} net-positive, ${dp.negativeDayCount} net-negative, ${dp.neutralDayCount} net-neutral`,
    `Recent 14-day average sentiment score: ${dp.recentAvgSentiment.toFixed(2)} (-1 very negative, 0 neutral, +1 very positive)`,
  ];
  if (dp.priceChangePercent != null) {
    lines.push(
      `Price change over this period: ${dp.priceChangePercent >= 0 ? "+" : ""}${dp.priceChangePercent.toFixed(2)}%`
    );
  }
  if (dp.priceRangeLow != null && dp.priceRangeHigh != null) {
    lines.push(
      `Price range over this period: $${dp.priceRangeLow.toFixed(2)} - $${dp.priceRangeHigh.toFixed(2)}`
    );
  }
  return lines.map((l) => `- ${l}`).join("\n");
}

function fallbackSections(
  companyName: string,
  ticker: string,
  dp: SeoDataPoints
): SeoPageSections {
  const trend =
    dp.recentAvgSentiment > 0.2 ? "leaning positive" : dp.recentAvgSentiment < -0.2 ? "leaning negative" : "mixed";
  const priceMove =
    dp.priceChangePercent != null
      ? `${dp.priceChangePercent >= 0 ? "gained" : "lost"} ${Math.abs(dp.priceChangePercent).toFixed(2)}%`
      : "moved";
  const priceMoveNoun =
    dp.priceChangePercent != null
      ? `${Math.abs(dp.priceChangePercent).toFixed(2)}% ${dp.priceChangePercent >= 0 ? "gain" : "loss"}`
      : "price action";
  const priceRangeText =
    dp.priceRangeLow != null && dp.priceRangeHigh != null
      ? `, trading between $${dp.priceRangeLow.toFixed(2)} and $${dp.priceRangeHigh.toFixed(2)}`
      : "";
  const majorityLabel =
    dp.positiveDayCount > dp.negativeDayCount && dp.positiveDayCount > dp.neutralDayCount
      ? "net-positive"
      : dp.negativeDayCount > dp.positiveDayCount && dp.negativeDayCount > dp.neutralDayCount
        ? "net-negative"
        : "net-neutral";

  // Used only when the LLM is unavailable, rate-limited, or comes back too
  // thin (see MIN_ACCEPTABLE_WORDS below) — deliberately built from real
  // per-ticker numbers rather than generic filler, so it's never a downgrade
  // from a rejected LLM response.
  return {
    intro: `${companyName} (${ticker}) is a widely-followed stock, and this page tracks how recent news coverage about it reads alongside its price movement. The analysis below draws on ${dp.totalArticles} news articles analyzed over the past ${dp.daysOfData} days, each classified as positive, negative, or neutral in reference to ${companyName} specifically. Over that window, ${ticker} has ${priceMove}${priceRangeText}. The goal of this page is to give a fast, data-grounded read on how the news cycle around ${ticker} compares to how the stock has actually traded, rather than relying on a single headline or data point in isolation.`,
    sentimentSummary: `Sentiment for ${ticker} is aggregated daily from recent news coverage. Across the ${dp.daysOfData}-day period tracked here, ${dp.positiveDayCount} days skewed net-positive, ${dp.negativeDayCount} skewed net-negative, and ${dp.neutralDayCount} were net-neutral overall, which puts the recent 14-day average sentiment score at ${dp.recentAvgSentiment.toFixed(2)} on a scale from -1 (very negative) to +1 (very positive) — ${trend}. The day-by-day mix has been predominantly ${majorityLabel} over the tracked window, which is one input into how the market may be pricing in near-term news for ${ticker}. Sentiment like this tends to move with product announcements, financial results, broader market conditions, and how a company is discussed relative to its sector peers. It won't always line up cleanly with price action day to day, since price also reflects factors sentiment scoring can't see, like trading volume, options positioning, and macro conditions unrelated to any single company.`,
    prediction: `Past sentiment trends and price history are not a guarantee of future performance for ${ticker}, and this page should be read as one input among several rather than a standalone signal. Sentiment can shift quickly around earnings releases, product news, leadership changes, or broader market moves, and a run of ${trend} coverage can reverse without much warning. The same is true of price: a ${dp.daysOfData}-day window is a short slice of a company's history, and the ${priceMoveNoun} over this period doesn't by itself predict what comes next. Anyone using this data to inform an investment decision should weigh it alongside fundamentals, valuation, and their own research, and consider talking to a licensed financial advisor before acting on it.`,
  };
}

/** Extracts the outermost {...} JSON object from an LLM response, tolerating stray prose/fences around it. */
function extractSections(raw: string): SeoPageSections | null {
  const cleaned = raw.replace(/^```json\s*/i, "").replace(/```$/, "").trim();
  const match = cleaned.match(/\{[\s\S]*\}/);
  try {
    const parsed = JSON.parse(match ? match[0] : cleaned);
    if (
      typeof parsed.intro === "string" &&
      typeof parsed.sentimentSummary === "string" &&
      typeof parsed.prediction === "string"
    ) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

function wordCount(sections: SeoPageSections): number {
  return Object.values(sections).join(" ").trim().split(/\s+/).filter(Boolean).length;
}

/** Below this, the LLM ignored the word-count instructions badly enough that
 * the richer, data-point-driven fallback text is the better result. */
const MIN_ACCEPTABLE_WORDS = 220;

function averageRecentSentiment(overlay: SentimentPricePoint[]): number {
  return (
    overlay
      .filter((p) => p.averageSentiment != null)
      .slice(-14)
      .reduce((sum, p, _, arr) => sum + (p.averageSentiment ?? 0) / arr.length, 0) || 0
  );
}

/** AI-generated intro/sentiment/prediction copy for a programmatic SEO sentiment page. */
export async function generateSeoPageSections(
  companyName: string,
  ticker: string,
  overlay: SentimentPricePoint[]
): Promise<SeoPageSections> {
  const dp = buildDataPoints(overlay);

  if (!llm7Client) return fallbackSections(companyName, ticker, dp);
  if (!seoGenerationLimiter.consume("global")) {
    console.warn(`[llm7] SEO sentiment generation rate-limited, using fallback for ${ticker}`);
    return fallbackSections(companyName, ticker, dp);
  }

  const prompt = `Company: ${companyName} (${ticker})

Data points (cite at least 3-5 of these specific numbers by name across the sections below — do not invent numbers not listed here):
${formatDataPointsForPrompt(dp)}

Write three sections for an SEO page about ${ticker} stock sentiment, as JSON with keys "intro", "sentimentSummary", "prediction". Each section has a MINIMUM word count AND minimum sentence count below — treat both as hard floors, not targets, and go over rather than under:
- "intro": at least 150 words, at least 7 sentences. Introduce ${companyName}, this page's purpose, and reference the article count and time window from the data points above.
- "sentimentSummary": at least 220 words, at least 9 sentences. Describe the recent sentiment trend in detail, citing the sentiment score, the positive/negative/neutral day breakdown, and price movement/range from the data points above. Discuss in general terms what tends to move sentiment for a company like this and how the trend compares to the price action.
- "prediction": at least 150 words, at least 7 sentences. Balanced, non-speculative commentary on what the sentiment and price data above might imply, what could change it, and a reminder that none of it guarantees future performance.

The combined total across all three sections must be at least 500 words — if you are unsure whether you've hit that, write more, not less.

Do not invent specific facts not provided in the data points above — no product names, dates, executives, earnings figures, or news events. When discussing possible drivers of sentiment, stay general (e.g. "product announcements" or "broader market conditions") rather than naming anything specific you were not given. Do not pad with filler or repeat the same sentence structure across sections; every sentence should add a distinct point.

Respond with ONLY the JSON object, no markdown fences.`;

  try {
    const response = await llm7Client.chat.completions.create({
      model: llm7Model,
      temperature: 0.4,
      max_tokens: 1400,
      messages: [
        {
          role: "system",
          content: "You are a financial content writer producing factual, balanced stock analysis copy.",
        },
        { role: "user", content: prompt },
      ],
    });

    const raw = response.choices[0]?.message?.content?.trim() ?? "";
    const sections = extractSections(raw);
    if (!sections) return fallbackSections(companyName, ticker, dp);

    if (wordCount(sections) < MIN_ACCEPTABLE_WORDS) {
      console.warn(
        `[llm7] SEO sentiment page for ${ticker} came back too thin (${wordCount(sections)} words), using fallback`
      );
      return fallbackSections(companyName, ticker, dp);
    }
    return sections;
  } catch (error) {
    console.error(`[llm7] SEO sentiment page generation failed for ${ticker}`, error);
    return fallbackSections(companyName, ticker, dp);
  }
}

export interface IndexWeeklyRecapInput {
  companyName: string;
  ticker: string;
  weekOfLabel: string;
  price: number | null;
  dayChangePercent: number | null;
  weekChangePercent: number | null;
  overlay: SentimentPricePoint[];
}

function fallbackIndexWeeklyRecapSections(input: IndexWeeklyRecapInput): SeoPageSections {
  const weekMove =
    input.weekChangePercent != null
      ? `${input.weekChangePercent >= 0 ? "gained" : "lost"} ${Math.abs(input.weekChangePercent).toFixed(2)}%`
      : "moved";
  return {
    intro: `The ${input.companyName} ${weekMove} for the week of ${input.weekOfLabel}. This page tracks its weekly performance alongside recent news sentiment.`,
    sentimentSummary: `News sentiment around the ${input.companyName} has been tracked over the trailing period, alongside daily price movement.`,
    prediction: `Weekly index performance and sentiment trends are not a guarantee of future results. Always do your own research before making investment decisions.`,
  };
}

/** AI-generated intro/recap/outlook copy for an index's weekly recap SEO page. */
export async function generateIndexWeeklyRecapSections(
  input: IndexWeeklyRecapInput
): Promise<SeoPageSections> {
  if (!llm7Client) return fallbackIndexWeeklyRecapSections(input);
  if (!seoGenerationLimiter.consume("global")) {
    console.warn(`[llm7] Index weekly recap generation rate-limited, using fallback for ${input.ticker}`);
    return fallbackIndexWeeklyRecapSections(input);
  }

  const recentAvg = averageRecentSentiment(input.overlay);

  const prompt = `Index: ${input.companyName} (${input.ticker})
Week of: ${input.weekOfLabel}
Latest price: ${input.price ?? "unknown"}
Most recent daily percent change: ${
    input.dayChangePercent != null ? input.dayChangePercent.toFixed(2) + "%" : "unknown"
  }
Percent change over the trailing week: ${
    input.weekChangePercent != null ? input.weekChangePercent.toFixed(2) + "%" : "unknown"
  }
Recent 14-day average news sentiment score (-1 = very negative, 0 = neutral, +1 = very positive): ${recentAvg.toFixed(2)}

Write three short sections for an SEO page recapping ${input.companyName}'s performance this week, as JSON with keys "intro", "sentimentSummary", "prediction":
- "intro": 2-3 sentences summarizing ${input.companyName}'s performance for the week of ${input.weekOfLabel}, referencing the percent change figures above.
- "sentimentSummary": 2-3 sentences describing the recent news sentiment trend based on the score above.
- "prediction": 2-3 sentences of balanced, non-speculative commentary noting that past performance and sentiment data are not a guarantee of future results.

Respond with ONLY the JSON object, no markdown fences.`;

  try {
    const response = await llm7Client.chat.completions.create({
      model: llm7Model,
      temperature: 0.4,
      max_tokens: 700,
      messages: [
        {
          role: "system",
          content: "You are a financial content writer producing factual, balanced market index recap copy.",
        },
        { role: "user", content: prompt },
      ],
    });

    const raw = response.choices[0]?.message?.content?.trim() ?? "";
    return extractSections(raw) ?? fallbackIndexWeeklyRecapSections(input);
  } catch (error) {
    console.error(`[llm7] Index weekly recap generation failed for ${input.ticker}`, error);
    return fallbackIndexWeeklyRecapSections(input);
  }
}
