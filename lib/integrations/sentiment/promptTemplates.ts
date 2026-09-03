export function buildSentimentPrompt(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): string {
  return `Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference to the company ${companyName}.

Title: ${articleTitle}
Description: ${articleDescription}

Do not assume anything not explicitly stated in the title or description.
Respond with only one word and nothing else: positive, negative, or neutral.`;
}

export type SentimentLabel = "positive" | "negative" | "neutral";

const LABELS: SentimentLabel[] = ["positive", "negative", "neutral"];

/**
 * Extracts a clean sentiment label from a raw LLM response — handles a
 * plain word, punctuation, a JSON-wrapped value, or free text containing
 * one of the three words.
 */
export function extractSentimentLabel(raw: string | null | undefined): SentimentLabel | null {
  if (!raw) return null;
  const cleaned = raw.trim().toLowerCase();

  const bareWord = cleaned.replace(/[^a-z]/g, "");
  if (LABELS.includes(bareWord as SentimentLabel)) {
    return bareWord as SentimentLabel;
  }

  try {
    const parsed = JSON.parse(raw);
    const candidate =
      typeof parsed === "string"
        ? parsed
        : parsed?.sentiment ?? parsed?.label ?? parsed?.value;
    if (typeof candidate === "string") {
      const normalized = candidate.trim().toLowerCase();
      if (LABELS.includes(normalized as SentimentLabel)) {
        return normalized as SentimentLabel;
      }
    }
  } catch {
    // not JSON, fall through to regex search
  }

  for (const label of LABELS) {
    if (new RegExp(`\\b${label}\\b`).test(cleaned)) return label;
  }

  return null;
}
