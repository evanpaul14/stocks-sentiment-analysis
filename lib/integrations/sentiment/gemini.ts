import { GoogleGenAI, ApiError } from "@google/genai";
import { buildSentimentPrompt, extractSentimentLabel, type SentimentLabel } from "./promptTemplates";
import { SlidingWindowRateLimiter, sleep } from "./rateLimiter";

// Gemma 3 (used by the old app) has been superseded — gemma-4-26b-a4b-it is
// the current smaller/faster variant, matching the old app's preference for
// a lighter model over the largest available one. Confirmed via a live
// ListModels call against this project's GOOGLE_API_KEY.
const MODEL = "gemma-4-26b-a4b-it";
const MAX_ATTEMPTS = 3;

const client = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });

const rateLimiter = new SlidingWindowRateLimiter(
  Number(process.env.GEMMA_MAX_CALLS_PER_MINUTE ?? 45),
  Number(process.env.GEMMA_RATE_WINDOW_SECONDS ?? 60) * 1000
);

const SLOW_RESPONSE_THRESHOLD_MS =
  Number(process.env.GEMMA_SENTIMENT_TIMEOUT_SECONDS ?? 3) * 1000;

export class ModelOverloadedError extends Error {
  constructor() {
    super("MODEL_OVERLOADED");
    this.name = "ModelOverloadedError";
  }
}

function isModelOverloaded(error: unknown): boolean {
  if (!(error instanceof ApiError)) return false;
  return error.status === 503 && error.message.toLowerCase().includes("overloaded");
}

/** Reads a "Retry-After ... seconds" style hint out of an error message. */
function extractRetryDelaySeconds(error: unknown): number | null {
  if (!(error instanceof Error)) return null;
  const match = error.message.match(/(\d+(?:\.\d+)?)\s*(seconds|secs|s)\b/i);
  if (!match) return null;
  const seconds = Number(match[1]);
  return Number.isFinite(seconds) ? seconds : null;
}

/**
 * Backup sentiment classifier (Gemini/Gemma). Retries up to MAX_ATTEMPTS,
 * respecting a rate limiter and any retry-delay hints. Throws
 * ModelOverloadedError (-> 503 at the route level) if the model stays
 * overloaded through all attempts with no usable result.
 */
export async function classifyWithGemini(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel> {
  const prompt = buildSentimentPrompt(companyName, articleTitle, articleDescription);

  let lastError: unknown = null;

  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    await rateLimiter.waitForSlot();
    const startedAt = performance.now();

    try {
      const response = await client.models.generateContent({
        model: MODEL,
        contents: prompt,
      });
      const elapsedMs = performance.now() - startedAt;
      const label = extractSentimentLabel(response.text) ?? "neutral";

      if (elapsedMs > SLOW_RESPONSE_THRESHOLD_MS) {
        console.warn(
          `[gemini] slow response (${elapsedMs.toFixed(0)}ms), accepting result without further retries`
        );
      }
      return label;
    } catch (error) {
      lastError = error;

      if (isModelOverloaded(error)) {
        console.warn("[gemini] model overloaded, aborting retries");
        break;
      }

      const retryDelaySeconds = extractRetryDelaySeconds(error);
      if (retryDelaySeconds != null) {
        await sleep(retryDelaySeconds * 1000);
        continue;
      }

      console.warn(`[gemini] attempt ${attempt + 1} failed`, error);
    }
  }

  if (isModelOverloaded(lastError)) {
    throw new ModelOverloadedError();
  }
  return "neutral";
}
