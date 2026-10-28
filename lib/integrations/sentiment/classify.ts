import { classifyWithCloudflare } from "./cloudflare";
import { classifyWithGemini } from "./gemini";
import { classifyBatchWithJev, classifyWithJev } from "./jev";
import type { SentimentLabel } from "./promptTemplates";

export type { SentimentLabel };
export { ModelOverloadedError } from "./gemini";

/** Jev primary, Cloudflare backup, Gemini as the last resort. */
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
  const cloudflareResult = await classifyWithCloudflare(
    companyName,
    articleTitle,
    articleDescription
  );
  if (cloudflareResult) return cloudflareResult;

  return classifyWithGemini(companyName, articleTitle, articleDescription);
}

export interface ArticleToClassify {
  title: string;
  description?: string | null;
}

/**
 * Classifies many articles about one company. Jev handles them in a single
 * batched request; any article Jev didn't answer (or all of them, if the batch
 * failed) goes through the existing per-article Cloudflare -> Gemini fallbacks,
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
