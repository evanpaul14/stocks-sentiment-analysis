import { classifyWithCloudflare } from "./cloudflare";
import { classifyWithGemini } from "./gemini";
import type { SentimentLabel } from "./promptTemplates";

export type { SentimentLabel };
export { ModelOverloadedError } from "./gemini";

/** Cloudflare primary, Gemini backup — mirrors the old app's analyze_sentiment(). */
export async function classifySentiment(
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
