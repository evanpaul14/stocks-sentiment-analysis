import type { SentimentLabel } from "./promptTemplates";

const ENDPOINT = "https://api.typesafe.ai/v1/systemone";
const DEFAULT_MODEL = "jev-latest";
const DEFAULT_TIMEOUT_SECONDS = 3;

function isEnabled(): boolean {
  return Boolean(process.env.TYPESAFE_API_KEY);
}

/**
 * Primary sentiment classifier (TypeSafe Jev). Returns null on any failure
 * (timeout, HTTP error, unexpected response) so the caller falls through to
 * the Cloudflare backup — no retries, fail fast.
 */
export async function classifyWithJev(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel | null> {
  if (!isEnabled()) return null;

  const timeoutMs =
    Number(process.env.TYPESAFE_TIMEOUT_SECONDS ?? DEFAULT_TIMEOUT_SECONDS) *
    1000;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.TYPESAFE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.TYPESAFE_MODEL ?? DEFAULT_MODEL,
        state: {
          company: companyName,
          title: articleTitle,
          ...(articleDescription ? { description: articleDescription } : {}),
        },
        questions: {
          sentiment: {
            type: "choice",
            instructions:
              "What is the sentiment of this article toward the price outlook of `company`?",
            criteria: {
              positive:
                "Good news for the stock: beats, upgrades, growth, buy recommendations, milestones",
              negative:
                "Bad news for the stock: drops, downgrades, sell recommendations, warnings, misses",
              neutral:
                "Routine or factual with no clear price impact, such as insider or fund holding changes, or mixed signals",
            },
          },
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `[jev] sentiment request failed with ${response.status}, falling back to Cloudflare`
      );
      return null;
    }

    const data = await response.json();
    const choice = data?.answers?.sentiment?.choice;
    return choice === "positive" || choice === "negative" || choice === "neutral"
      ? choice
      : null;
  } catch (error) {
    console.warn("[jev] sentiment request errored, falling back to Cloudflare", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
