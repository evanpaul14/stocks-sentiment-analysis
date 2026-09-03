import { llm7Client, llm7Model } from "./client";
import type { SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";

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

/** AI-generated intro/sentiment/prediction copy for a programmatic SEO sentiment page. */
export async function generateSeoPageSections(
  companyName: string,
  ticker: string,
  overlay: SentimentPricePoint[]
): Promise<SeoPageSections> {
  if (!llm7Client) return fallbackSections(companyName, ticker);

  const recentAvg =
    overlay.filter((p) => p.averageSentiment != null).slice(-14).reduce(
      (sum, p, _, arr) => sum + (p.averageSentiment ?? 0) / arr.length,
      0
    ) || 0;

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
      max_tokens: 400,
      messages: [
        {
          role: "system",
          content: "You are a financial content writer producing factual, balanced stock analysis copy.",
        },
        { role: "user", content: prompt },
      ],
    });

    const raw = response.choices[0]?.message?.content?.trim() ?? "";
    const cleaned = raw.replace(/^```json\s*/i, "").replace(/```$/, "");
    const parsed = JSON.parse(cleaned);

    if (
      typeof parsed.intro === "string" &&
      typeof parsed.sentimentSummary === "string" &&
      typeof parsed.prediction === "string"
    ) {
      return parsed;
    }
    return fallbackSections(companyName, ticker);
  } catch (error) {
    console.error(`[llm7] SEO sentiment page generation failed for ${ticker}`, error);
    return fallbackSections(companyName, ticker);
  }
}
