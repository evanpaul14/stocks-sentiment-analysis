import { llm7Client, llm7Model } from "./client";
import { fetchFinnhubCompanyNews, type Headline } from "@/lib/integrations/news/finnhub";
import { getNewsArticles } from "@/lib/integrations/news/googleNews";
import { TtlCache } from "@/lib/cache/memory";

const MOVEMENT_THRESHOLD_PERCENT = 3;

export interface MovementInsight {
  summary: string;
  source: "Finnhub" | "news";
}

const CACHE_TTL_MS =
  Number(process.env.MOVEMENT_INSIGHT_CACHE_TTL_SECONDS ?? 900) * 1000;
const insightCache = new TtlCache<MovementInsight | null>(CACHE_TTL_MS);

async function getCatalystHeadlines(
  symbol: string
): Promise<{ headlines: Headline[]; source: "Finnhub" | "news" }> {
  const finnhubHeadlines = await fetchFinnhubCompanyNews(symbol);
  if (finnhubHeadlines.length > 0) {
    return { headlines: finnhubHeadlines, source: "Finnhub" };
  }

  const articles = await getNewsArticles(symbol, 6);
  return {
    headlines: articles.map((a) => ({
      headline: a.title,
      summary: a.description,
      source: a.source,
      url: a.link,
      publishedAt: a.publishedAt ?? "",
    })),
    source: "news",
  };
}

function fallbackSummary(
  companyName: string,
  changePercent: number,
  headlines: Headline[]
): string {
  const direction = changePercent >= 0 ? "higher" : "lower";
  const base = `${companyName} is trading ${direction} by roughly ${Math.abs(
    changePercent
  ).toFixed(2)}% today.`;
  if (headlines.length === 0) {
    return `${base} No fresh headlines are available to explain the move.`;
  }
  const titles = headlines.slice(0, 3).map((h) => h.headline).join("; ");
  return `${base} Related headlines: ${titles}.`;
}

async function summarizeWithLlm7(
  symbol: string,
  companyName: string,
  changePercent: number,
  headlines: Headline[]
): Promise<string | null> {
  if (!llm7Client || headlines.length === 0) return null;

  const direction = changePercent >= 0 ? "up" : "down";
  const bullets = headlines
    .map(
      (h, i) =>
        `${i + 1}. ${h.headline} (${h.source}) on ${h.publishedAt}: ${h.summary}`
    )
    .join("\n");

  const userPrompt = `${companyName} (${symbol}) is trading ${direction} ${Math.abs(
    changePercent
  ).toFixed(2)}% today.

Catalyst headlines:
${bullets}

Write a 2-3 sentence, no-speculation explanation of the move. Refer to the company by its ticker (${symbol}), not its full name. Do not reference "the headlines" or "the article" directly, and do not speculate beyond what the headlines state. If the headlines do not explain the move, clearly state that. Do not specifically mention the stock price or percentage change in the summary. Do not include a headline or summary section indicator, just give the summary.`;

  try {
    const response = await llm7Client.chat.completions.create({
      model: llm7Model,
      temperature: 0.3,
      max_tokens: 220,
      messages: [
        {
          role: "system",
          content:
            "You are a sharp markets reporter who explains price action using headlines.",
        },
        { role: "user", content: userPrompt },
      ],
    });
    return response.choices[0]?.message?.content?.trim() ?? null;
  } catch (error) {
    console.error(`[llm7] movement insight failed for ${symbol}`, error);
    return null;
  }
}

/**
 * Only triggers for stocks with a +/-3% price move. Cached per symbol so
 * reloading the same stock page within the TTL doesn't re-hit the LLM.
 */
export async function buildMovementInsight(
  symbol: string,
  companyName: string,
  changePercent: number
): Promise<MovementInsight | null> {
  if (Math.abs(changePercent) < MOVEMENT_THRESHOLD_PERCENT) return null;

  return insightCache.getOrCompute(symbol.toUpperCase(), async () => {
    const { headlines, source } = await getCatalystHeadlines(symbol);

    const llmSummary = await summarizeWithLlm7(symbol, companyName, changePercent, headlines);
    const summary = llmSummary ?? fallbackSummary(companyName, changePercent, headlines);

    return { summary, source };
  });
}
