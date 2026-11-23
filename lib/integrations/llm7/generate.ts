import { llm7Client, llm7Model } from "./client";

const DEFAULT_CLOUDFLARE_TEXT_MODEL = "@cf/google/gemma-4-26b-a4b-it";
const DEFAULT_CLOUDFLARE_LONG_TEXT_MODEL = "@cf/google/gemma-4-26b-a4b-it";
const DEFAULT_CLOUDFLARE_TEXT_TIMEOUT_SECONDS = 30;
const DEFAULT_LLM7_TIMEOUT_SECONDS = 20;

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
  /**
   * How long to wait on LLM7 before failing over to Cloudflare. Latency-sensitive
   * callers (user-facing pages) pass a small value; batch jobs use the default.
   */
  llm7TimeoutMs?: number;
  /** Try Cloudflare first and fall back to LLM7 (default is LLM7 first). */
  preferCloudflare?: boolean;
  /** Overrides how long to wait on the Cloudflare fallback. */
  cloudflareTimeoutMs?: number;
}

/** Epoch ms until which LLM7 is skipped, set when it answers 429 with a wait time. */
let llm7BlockedUntil = 0;

/** Reads the wait from a 429: the Retry-After header, else "retry in N seconds" in the message. */
function parseRetryAfterMs(error: unknown): number | null {
  const err = error as { status?: number; message?: string; headers?: Headers | Record<string, string> };
  if (err?.status !== 429) return null;
  const h = err.headers;
  const header =
    h && typeof (h as Headers).get === "function"
      ? (h as Headers).get("retry-after")
      : (h as Record<string, string> | undefined)?.["retry-after"];
  const fromHeader = header ? Number(header) : NaN;
  if (Number.isFinite(fromHeader) && fromHeader > 0) return fromHeader * 1000;
  const match = err.message?.match(/retry in (\d+(?:\.\d+)?)\s*seconds?/i);
  return match ? Number(match[1]) * 1000 : null;
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
    opts.cloudflareTimeoutMs ??
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
        // Gemma is a reasoning model: without this it spends max_tokens on
        // reasoning_content and returns empty content. Ignored by non-reasoning models.
        chat_template_kwargs: { enable_thinking: false },
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      console.error(`[cloudflare] ${opts.label} failed with ${response.status}`);
      return null;
    }
    const data = await response.json();
    // Older models return result.response; newer ones use OpenAI-style choices.
    const raw =
      data?.result?.response ??
      data?.result?.choices?.[0]?.message?.content ??
      data?.result?.output ??
      null;
    return typeof raw === "string" && raw.trim() ? raw.trim() : null;
  } catch (error) {
    console.error(`[cloudflare] ${opts.label} errored`, error);
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

async function generateWithLlm7(opts: GenerateTextOptions): Promise<string | null> {
  if (!llm7Client || Date.now() < llm7BlockedUntil) return null;
  try {
    // maxRetries: 0 — the SDK's default retries would stack on top of the
    // timeout before we ever reach the fallback provider.
    const response = await llm7Client.chat.completions.create(
      {
        model: llm7Model,
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        messages: opts.messages,
      },
      {
        timeout: opts.llm7TimeoutMs ?? DEFAULT_LLM7_TIMEOUT_SECONDS * 1000,
        maxRetries: 0,
      }
    );
    const content = response.choices[0]?.message?.content?.trim();
    if (content) return content;
    console.error(`[llm7] ${opts.label} returned empty content`);
  } catch (error) {
    const waitMs = parseRetryAfterMs(error);
    if (waitMs) {
      llm7BlockedUntil = Date.now() + waitMs;
      console.warn(
        `[llm7] rate limited; skipping LLM7 for ${Math.round(waitMs / 1000)}s (until ${new Date(llm7BlockedUntil).toISOString()})`
      );
    } else {
      console.error(`[llm7] ${opts.label} failed`, error);
    }
  }
  return null;
}

/**
 * Generates text with LLM7 then Cloudflare Workers AI (or the reverse when
 * `preferCloudflare` is set), using the second provider if the first errors or
 * returns nothing. Returns null if no provider produced text, so callers can
 * use their own templated fallback.
 */
export async function generateText(opts: GenerateTextOptions): Promise<string | null> {
  const cloudflare = async () => (cloudflareEnabled() ? generateWithCloudflare(opts) : null);
  const [first, second] = opts.preferCloudflare
    ? [cloudflare, () => generateWithLlm7(opts)]
    : [() => generateWithLlm7(opts), cloudflare];
  const result = await first();
  if (result) return result;
  console.warn(`[llm] falling back to second provider for ${opts.label}`);
  return second();
}
