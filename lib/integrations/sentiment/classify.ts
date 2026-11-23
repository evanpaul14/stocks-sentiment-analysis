import { classifyWithCloudflare } from "./cloudflare";
import { classifyWithGeminiDecision } from "./geminiDecision";
import { classifyBatchWithJev, classifyWithJev } from "./jev";
import type { SentimentLabel } from "./promptTemplates";

export type { SentimentLabel };
/** Thrown when every classifier in the chain failed to produce a label. */
export class ModelOverloadedError extends Error {
  constructor() {
    super("MODEL_OVERLOADED");
    this.name = "ModelOverloadedError";
  }
}

/** Jev primary, Gemini Decision 1 backup, Cloudflare Gemma as the last resort. */
export async function classifySentiment(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel> {
  const jevResult = await classifyWithJev(
    companyName,
    articleTitle,
    articleDescription
  );
  if (jevResult) return jevResult;

  return classifyWithFallbacks(companyName, articleTitle, articleDescription);
}

async function classifyWithFallbacks(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel> {
  const geminiResult = await classifyWithGeminiDecision(
    companyName,
    articleTitle,
    articleDescription
  );
  if (geminiResult) return geminiResult;

  const cloudflareResult = await classifyWithCloudflare(
    companyName,
    articleTitle,
    articleDescription
  );
  if (cloudflareResult) return cloudflareResult;

  // Nothing answered: fail rather than store a made-up "neutral".
  throw new ModelOverloadedError();
}

export interface ArticleToClassify {
  title: string;
  description?: string | null;
}

/**
 * Classifies many articles about one company. Jev handles them in a single
 * batched request; any article Jev didn't answer (or all of them, if the batch
 * failed) goes through the existing per-article Gemini Decision -> Cloudflare fallbacks,
 * one request each, without retrying Jev. Results are settled per article so one
 * failed fallback doesn't lose the rest.
 */
export async function classifySentimentBatch(
  companyName: string,
  articles: ArticleToClassify[]
): Promise<PromiseSettledResult<SentimentLabel>[]> {
  const jevResults = await classifyBatchWithJev(
    companyName,
    articles.map((a) => ({ title: a.title, description: a.description ?? "" }))
  );

  return Promise.allSettled(
    articles.map(
      async (article, i) =>
        jevResults?.[i] ??
        classifyWithFallbacks(companyName, article.title, article.description ?? "")
    )
  );
}
