import type { SentimentLabel } from "./promptTemplates";
import { SlidingWindowRateLimiter } from "./rateLimiter";

const ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions";
const DEFAULT_MODEL = "gemini-decision-1";
const DEFAULT_TIMEOUT_SECONDS = 3;

// Same criteria as the Jev classifier so the two are interchangeable.
const SENTIMENT_CRITERIA = {
  positive:
    "Good news for the stock: beats, upgrades, growth, buy recommendations, milestones",
  negative:
    "Bad news for the stock: drops, downgrades, sell recommendations, warnings, misses",
  neutral:
    "Routine or factual with no clear price impact, such as insider or fund holding changes, or mixed signals",
};

// Free tier allows 200 requests/minute.
const rateLimiter = new SlidingWindowRateLimiter(
  Number(process.env.GEMINI_DECISION_MAX_CALLS_PER_MINUTE ?? 200),
  60_000
);

function isEnabled(): boolean {
  return Boolean(process.env.GOOGLE_API_KEY);
}

/**
 * Second-line sentiment classifier (Gemini Decision 1 via the Decisions API).
 * Returns null on any failure (timeout, HTTP error, unexpected response) so
 * the caller falls through to the Cloudflare backup — no retries, fail fast.
 */
export async function classifyWithGeminiDecision(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel | null> {
  if (!isEnabled()) return null;

  const timeoutMs =
    Number(process.env.GEMINI_DECISION_TIMEOUT_SECONDS ?? DEFAULT_TIMEOUT_SECONDS) *
    1000;

  await rateLimiter.waitForSlot();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        "x-goog-api-key": process.env.GOOGLE_API_KEY!,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.GEMINI_DECISION_MODEL ?? DEFAULT_MODEL,
        input: `Company: ${companyName}\nTitle: ${articleTitle}${
          articleDescription ? `\nDescription: ${articleDescription}` : ""
        }`,
        questions: {
          sentiment: {
            type: "choice",
            instructions: `What is the sentiment of this article toward the price outlook of ${companyName}?`,
            criteria: SENTIMENT_CRITERIA,
          },
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `[gemini-decision] sentiment request failed with ${response.status}, falling back to Cloudflare`
      );
      return null;
    }

    const data = await response.json();
    const choice = data?.answers?.sentiment?.choice;
    return choice === "positive" || choice === "negative" || choice === "neutral"
      ? choice
      : null;
  } catch (error) {
    console.warn("[gemini-decision] sentiment request errored, falling back to Cloudflare", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
