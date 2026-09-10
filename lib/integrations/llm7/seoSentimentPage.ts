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

function fallbackSections(companyName: string, ticker: string): SeoPageSections {
  return {
    intro: `${companyName} (${ticker}) is a widely-followed stock. This page tracks recent news sentiment alongside price movement over the last 90 days.`,
    sentimentSummary: `Sentiment data for ${ticker} is aggregated from recent news coverage, classified as positive, negative, or neutral.`,
    prediction: `Past sentiment trends are not a guarantee of future price movement. Always do your own research before making investment decisions.`,
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
  if (!llm7Client) return fallbackSections(companyName, ticker);
  if (!seoGenerationLimiter.consume("global")) {
    console.warn(`[llm7] SEO sentiment generation rate-limited, using fallback for ${ticker}`);
    return fallbackSections(companyName, ticker);
  }

  const recentAvg = averageRecentSentiment(overlay);

  const prompt = `Company: ${companyName} (${ticker})
Recent 14-day average sentiment score (-1 = very negative, 0 = neutral, +1 = very positive): ${recentAvg.toFixed(2)}
Data points available: ${overlay.length} days

Write three short sections for an SEO page about ${ticker} stock sentiment, as JSON with keys "intro", "sentimentSummary", "prediction":
- "intro": 2-3 sentences introducing ${companyName} and this page's purpose.
- "sentimentSummary": 2-3 sentences describing the recent sentiment trend based on the score above.
- "prediction": 2-3 sentences of balanced, non-speculative commentary noting that sentiment data is not a guarantee of future performance.

Respond with ONLY the JSON object, no markdown fences.`;

  try {
    const response = await llm7Client.chat.completions.create({
      model: llm7Model,
      temperature: 0.4,
      max_tokens: 700,
      messages: [
        {
          role: "system",
          content: "You are a financial content writer producing factual, balanced stock analysis copy.",
        },
        { role: "user", content: prompt },
      ],
    });

    const raw = response.choices[0]?.message?.content?.trim() ?? "";
    return extractSections(raw) ?? fallbackSections(companyName, ticker);
  } catch (error) {
    console.error(`[llm7] SEO sentiment page generation failed for ${ticker}`, error);
    return fallbackSections(companyName, ticker);
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
