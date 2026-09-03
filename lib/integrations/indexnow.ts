const INDEXNOW_URL = "https://api.indexnow.org/indexnow";

/**
 * Best-effort, fire-and-forget push to IndexNow (Bing/Yandex/Naver/Seznam)
 * on publish/update of market-summary or blog URLs. Never throws.
 */
export async function pingIndexNow(urls: string[]): Promise<void> {
  const key = process.env.INDEXNOW_KEY;
  const baseUrl = process.env.SITE_BASE_URL;
  if (!key || !baseUrl || urls.length === 0) return;

  try {
    const host = new URL(baseUrl).host;
    await fetch(INDEXNOW_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        host,
        key,
        keyLocation: `${baseUrl}/${key}.txt`,
        urlList: urls,
      }),
      signal: AbortSignal.timeout(5000),
    });
  } catch (error) {
    console.warn("[indexnow] ping failed", error);
  }
}
