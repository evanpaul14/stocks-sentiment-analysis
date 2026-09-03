import { buildSentimentPrompt, extractSentimentLabel, type SentimentLabel } from "./promptTemplates";

const DEFAULT_MODEL = "@cf/meta/llama-3.2-3b-instruct";
const DEFAULT_TIMEOUT_SECONDS = 3;

function isEnabled(): boolean {
  return Boolean(
    process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN
  );
}

/**
 * Primary sentiment classifier. Returns null on any failure (timeout, HTTP
 * error, unparseable response) so the caller falls through to the Gemini
 * backup — mirrors the old app's "no retries, fail fast" behavior.
 */
export async function classifyWithCloudflare(
  companyName: string,
  articleTitle: string,
  articleDescription: string
): Promise<SentimentLabel | null> {
  if (!isEnabled()) return null;

  const model = process.env.CLOUDFLARE_SENTIMENT_MODEL ?? DEFAULT_MODEL;
  const timeoutMs =
    Number(process.env.CLOUDFLARE_TIMEOUT_SECONDS ?? DEFAULT_TIMEOUT_SECONDS) *
    1000;
  const url = `https://api.cloudflare.com/client/v4/accounts/${process.env.CLOUDFLARE_ACCOUNT_ID}/ai/run/${model}`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.CLOUDFLARE_API_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        messages: [
          { role: "system", content: "You are a stock expert" },
          {
            role: "user",
            content: buildSentimentPrompt(
              companyName,
              articleTitle,
              articleDescription
            ),
          },
        ],
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      console.warn(
        `[cloudflare] sentiment request failed with ${response.status}, falling back to Gemini`
      );
      return null;
    }

    const data = await response.json();
    const raw = data?.result?.response ?? data?.result?.output ?? null;
    return extractSentimentLabel(typeof raw === "string" ? raw : null);
  } catch (error) {
    console.warn("[cloudflare] sentiment request errored, falling back to Gemini", error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}
