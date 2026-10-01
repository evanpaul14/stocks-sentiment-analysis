import { llm7Client, llm7Model } from "./client";

const DEFAULT_CLOUDFLARE_TEXT_MODEL = "@cf/meta/llama-3.2-3b-instruct";
const DEFAULT_CLOUDFLARE_LONG_TEXT_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const DEFAULT_CLOUDFLARE_TEXT_TIMEOUT_SECONDS = 30;

export interface ChatMessage {
  role: "system" | "user";
  content: string;
}

export interface GenerateTextOptions {
  messages: ChatMessage[];
  temperature: number;
  maxTokens: number;
  /** Used in log lines, e.g. "movement insight for AAPL". */
  label: string;
  /** Longer outputs use the larger Cloudflare fallback model. */
  long?: boolean;
}

function cloudflareEnabled(): boolean {
  return Boolean(process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN);
}

/** True if at least one text-generation provider (LLM7 or Cloudflare) is configured. */
export function llmAvailable(): boolean {
  return Boolean(llm7Client) || cloudflareEnabled();
}

async function generateWithCloudflare(opts: GenerateTextOptions): Promise<string | null> {
  const model = opts.long
    ? (process.env.CLOUDFLARE_LONG_TEXT_MODEL ?? DEFAULT_CLOUDFLARE_LONG_TEXT_MODEL)
    : (process.env.CLOUDFLARE_TEXT_MODEL ?? DEFAULT_CLOUDFLARE_TEXT_MODEL);
  const timeoutMs =
    Number(process.env.CLOUDFLARE_TEXT_TIMEOUT_SECONDS ?? DEFAULT_CLOUDFLARE_TEXT_TIMEOUT_SECONDS) *
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
        messages: opts.messages,
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      console.error(`[cloudflare] ${opts.label} failed with ${response.status}`);
      return null;
    }
    const data = await response.json();
    const raw = data?.result?.response ?? data?.result?.output ?? null;
    return typeof raw === "string" && raw.trim() ? raw.trim() : null;
  } catch (error) {
    console.error(`[cloudflare] ${opts.label} errored`, error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Generates text with LLM7, falling back to Cloudflare Workers AI if LLM7 errors
 * or returns nothing. Returns null if no provider produced text, so callers can
 * use their own templated fallback.
 */
export async function generateText(opts: GenerateTextOptions): Promise<string | null> {
  if (llm7Client) {
    try {
      const response = await llm7Client.chat.completions.create({
        model: llm7Model,
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        messages: opts.messages,
      });
      const content = response.choices[0]?.message?.content?.trim();
      if (content) return content;
      console.error(`[llm7] ${opts.label} returned empty content`);
    } catch (error) {
      console.error(`[llm7] ${opts.label} failed`, error);
    }
  }

  if (!cloudflareEnabled()) return null;
  console.warn(`[llm7] falling back to Cloudflare for ${opts.label}`);
  return generateWithCloudflare(opts);
}
