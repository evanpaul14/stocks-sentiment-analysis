import type { SentimentLabel } from "./promptTemplates";

const ENDPOINT = "https://api.typesafe.ai/v1/systemone";
const DEFAULT_MODEL = "jev-latest";
const DEFAULT_TIMEOUT_SECONDS = 3;

const SENTIMENT_CRITERIA = {
  positive:
    "Good news for the stock: beats, upgrades, growth, buy recommendations, milestones",
  negative:
    "Bad news for the stock: drops, downgrades, sell recommendations, warnings, misses",
  neutral:
    "Routine or factual with no clear price impact, such as insider or fund holding changes, or mixed signals",
};

function isEnabled(): boolean {
  return Boolean(process.env.TYPESAFE_API_KEY);
}

/**
 * Primary sentiment classifier (TypeSafe Jev). Returns null on any failure
 * (timeout, HTTP error, unexpected response) so the caller falls through to
 * the Gemini Decision backup — no retries, fail fast.
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
            criteria: SENTIMENT_CRITERIA,
          },
        },
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `[jev] sentiment request failed with ${response.status}, falling back to Gemini Decision`
      );
      return null;
    }

    const data = await response.json();
    const choice = data?.answers?.sentiment?.choice;
    return choice === "positive" || choice === "negative" || choice === "neutral"
      ? choice
      : null;
  } catch (error) {
    console.warn("[jev] sentiment request errored, falling back to Gemini Decision", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

export interface JevArticle {
  title: string;
  description?: string;
}

/**
 * Classifies several articles about one company in a single Jev request: the
 * company goes in `state` once, and each article's title/description goes in
 * its own choice question, so articles can't influence each other's answers.
 * Returns one entry per input article (null where Jev gave no usable answer),
 * or null for the whole batch on any request failure, so the caller can fall
 * back to the per-article Gemini Decision/Cloudflare path. Questions are answered
 * independently, so a missing answer only affects its own article.
 */
export async function classifyBatchWithJev(
  companyName: string,
  articles: JevArticle[]
): Promise<(SentimentLabel | null)[] | null> {
  if (!isEnabled() || articles.length === 0) return null;

  const timeoutMs =
    Number(process.env.TYPESAFE_TIMEOUT_SECONDS ?? DEFAULT_TIMEOUT_SECONDS) *
    1000;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  // Only the company is shared; each article lives inside its own question so
  // one article's content can't sway the answer for another.
  const state = { company: companyName };
  const questions: Record<string, unknown> = {};
  articles.forEach((article, i) => {
    const articleText = article.description
      ? `Title: ${article.title}\nDescription: ${article.description}`
      : `Title: ${article.title}`;
    questions[`article_${i}`] = {
      type: "choice",
      instructions: `What is the sentiment of this article toward the price outlook of \`company\`?\n\n${articleText}`,
      criteria: SENTIMENT_CRITERIA,
    };
  });

  try {
    const response = await fetch(ENDPOINT, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.TYPESAFE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.TYPESAFE_MODEL ?? DEFAULT_MODEL,
        state,
        questions,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `[jev] batch sentiment request failed with ${response.status}, falling back to per-article classification`
      );
      return null;
    }

    const data = await response.json();
    return articles.map((_, i) => {
      const choice = data?.answers?.[`article_${i}`]?.choice;
      return choice === "positive" || choice === "negative" || choice === "neutral"
        ? choice
        : null;
    });
  } catch (error) {
    console.warn("[jev] batch sentiment request errored, falling back to per-article classification", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
