import { llm7Client, llm7Model } from "./client";
import type { IndexSnapshot } from "@/lib/integrations/yahoo/indices";
import type { NewsArticle } from "@/lib/integrations/news/googleNews";

function formatIndexLine(index: IndexSnapshot): string {
  const change =
    index.changePercent != null
      ? `${index.changePercent >= 0 ? "+" : ""}${index.changePercent.toFixed(2)}%`
      : "n/a";
  return `${index.name}: ${change} today`;
}

const PREAMBLE_LINE_PATTERN =
  /^(let me\b|okay,?\b|sure,?\b|here('|')s\b|i('|')ll\b|i will\b|certainly,?\b).*$/i;

/** Some LLM7 models leak a chain-of-thought preamble ("Let me analyze the data...")
 * before the actual summary despite the prompt's "no preamble" rule. Strip any
 * leading lines that look like that instead of the real briefing. */
function stripLeadingPreamble(text: string): string {
  const lines = text.split("\n");
  while (lines.length > 0 && PREAMBLE_LINE_PATTERN.test(lines[0].trim())) {
    lines.shift();
    if (lines[0]?.trim() === "") lines.shift();
  }
  return lines.join("\n").trim();
}

function fallbackMarketSummaryText(
  dateLabel: string,
  indexes: IndexSnapshot[],
  headlines: NewsArticle[]
): string {
  const indexLines = indexes.map(formatIndexLine).join(". ");
  const headlineTitles = headlines
    .slice(0, 3)
    .map((h) => h.title)
    .join("; ");
  const headlinePart = headlineTitles ? ` Top headlines: ${headlineTitles}.` : "";
  return `Markets wrap for ${dateLabel}. ${indexLines}.${headlinePart}`;
}

/** LLM-generated market summary, falling back to a templated summary if LLM7 is unavailable. */
export async function generateMarketSummaryText(
  dateLabel: string,
  indexes: IndexSnapshot[],
  headlines: NewsArticle[]
): Promise<string> {
  if (!llm7Client) return fallbackMarketSummaryText(dateLabel, indexes, headlines);

  const indexLines = indexes.map(formatIndexLine).join("\n") || "No index data available.";
  const headlineLines =
    headlines.map((h) => `- ${h.title} (${h.source}): ${h.description}`).join("\n") ||
    "No major headlines captured.";

  const userPrompt =
    `Date: ${dateLabel}\n` +
    "Today's index performances:\n" +
    `${indexLines}\n\n` +
    "Headline digest:\n" +
    `${headlineLines}\n` +
    "Task: Write a two paragraph summary of today's market session. " +
    "Explain the overall tone (risk-on/off, rally, selloff, mixed), " +
    "the most significant macro or sector drivers, and any notable themes from the headlines." +
    "Be as detailed as possible with information given while still being concise and easy to read." +
    "Avoid hype and keep it factual.\n\n" +
    "Rules:\n" +
    "- Use the index data to anchor the narrative (e.g. whether moves were broad or uneven)\n" +
    "- Incorporate specific details and figures from the headlines where relevant\n" +
    "- Do not reference individual articles or sources by name\n" +
    "- Do not speculate or invent facts not supported by the headlines\n" +
    "- No headers, labels, or preamble — output the summary only\n";

  try {
    const response = await llm7Client.chat.completions.create({
      model: llm7Model,
      temperature: 0.3,
      max_tokens: 350,
      messages: [
        {
          role: "system",
          content:
            "You are a financial markets writer producing a concise end-of-day briefing.",
        },
        { role: "user", content: userPrompt },
      ],
    });
    const content = response.choices[0]?.message?.content?.trim();
    if (!content) return fallbackMarketSummaryText(dateLabel, indexes, headlines);
    return stripLeadingPreamble(content);
  } catch (error) {
    console.error("[llm7] market summary generation failed", error);
    return fallbackMarketSummaryText(dateLabel, indexes, headlines);
  }
}
