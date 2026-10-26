import { classifyWithCloudflare } from "./cloudflare";
import { classifyWithGemini } from "./gemini";
import { classifyWithJev } from "./jev";
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

  const cloudflareResult = await classifyWithCloudflare(
    companyName,
    articleTitle,
    articleDescription
  );
  if (cloudflareResult) return cloudflareResult;

  return classifyWithGemini(companyName, articleTitle, articleDescription);
}
