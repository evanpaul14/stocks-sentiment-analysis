import { createHash } from "node:crypto";
import * as articleImage from "@/lib/db/queries/articleImage";

const RANDOM_PHOTO_URL = "https://api.unsplash.com/photos/random";

function isEnabled(): boolean {
  return Boolean(process.env.UNSPLASH_ACCESS_KEY);
}

function authHeaders() {
  return { Authorization: `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}` };
}

export function hashCacheKey(input: string): string {
  return createHash("sha256").update(input).digest("hex");
}

function sanitizeQuery(text: string): string {
  return text
    .replace(/[^a-zA-Z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .slice(0, 6)
    .join(" ");
}

function appendUtmParams(url: string): string {
  const appName = process.env.UNSPLASH_APP_NAME ?? "stocks-sentiment-analysis";
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}utm_source=${encodeURIComponent(appName)}&utm_medium=referral`;
}

interface UnsplashPhoto {
  id: string;
  urls: { regular?: string; full?: string; small?: string; thumb?: string };
  description?: string | null;
  alt_description?: string | null;
  user: { name: string; username: string; links?: { html?: string } };
  links: { html: string; download_location: string };
}

async function requestUnsplashPhoto(query: string): Promise<UnsplashPhoto | null> {
  const url = new URL(RANDOM_PHOTO_URL);
  url.searchParams.set("query", query);
  url.searchParams.set("orientation", "landscape");
  url.searchParams.set("count", "1");
  url.searchParams.set("content_filter", "high");

  const response = await fetch(url, {
    headers: authHeaders(),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) return null;

  const data = await response.json();
  const photo = Array.isArray(data) ? data[0] : data;
  return photo ?? null;
}

async function registerDownload(photo: UnsplashPhoto): Promise<void> {
  try {
    const url = new URL(photo.links.download_location);
    url.searchParams.set("client_id", process.env.UNSPLASH_ACCESS_KEY!);
    await fetch(url, { headers: authHeaders(), signal: AbortSignal.timeout(5000) });
  } catch {
    // best-effort per Unsplash API guidelines, never block on this
  }
}

export interface CachedImage {
  imageUrl: string;
  thumbnailUrl: string | null;
  photographerName: string | null;
  photographerProfileUrl: string | null;
}

/** Fetches (or reuses a cached) Unsplash photo for a cache key + search query. */
export async function getOrFetchUnsplashImage(
  cacheKey: string,
  query: string
): Promise<CachedImage | null> {
  const existing = await articleImage.getByKey(cacheKey);
  if (existing) {
    const attribution = existing.attributionJson ? JSON.parse(existing.attributionJson) : {};
    return {
      imageUrl: existing.url,
      thumbnailUrl: existing.thumbnailUrl,
      photographerName: attribution.photographerName ?? null,
      photographerProfileUrl: attribution.photographerProfileUrl ?? null,
    };
  }

  if (!isEnabled()) return null;

  const sanitizedQuery = sanitizeQuery(query) || (process.env.UNSPLASH_DEFAULT_QUERY ?? "stock market");
  const photo = await requestUnsplashPhoto(sanitizedQuery);
  if (!photo) return null;

  void registerDownload(photo);

  const imageUrl = photo.urls.regular ?? photo.urls.full ?? "";
  const photographerProfileUrl = appendUtmParams(
    photo.user.links?.html ?? `https://unsplash.com/@${photo.user.username}`
  );

  const saved = await articleImage.insertIfAbsent({
    key: cacheKey,
    unsplashId: photo.id,
    url: imageUrl,
    thumbnailUrl: photo.urls.small ?? photo.urls.thumb ?? null,
    downloadLocation: photo.links.download_location,
    attributionJson: JSON.stringify({
      photographerName: photo.user.name,
      photographerUsername: photo.user.username,
      photographerProfileUrl,
    }),
  });

  if (!saved) return null;

  return {
    imageUrl: saved.url,
    thumbnailUrl: saved.thumbnailUrl,
    photographerName: photo.user.name,
    photographerProfileUrl,
  };
}
