import { cache } from "react";
import * as sentimentPageCache from "@/lib/db/queries/sentimentPageCache";
import { getSentimentPriceOverlay, type SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";
import {
  generateIndexWeeklyRecapSections,
  generateSeoPageSections,
  type SeoPageSections,
} from "@/lib/integrations/llm7/seoSentimentPage";
import { getMarketIndexSnapshots } from "@/lib/integrations/yahoo/indices";
import { getOrFetchUnsplashImage, hashCacheKey } from "@/lib/integrations/unsplash";
import { formatWeekOfLabel } from "@/lib/utils/dates";
import {
  SEO_SENTIMENT_COMPANIES,
  companySlug,
  findSeoCompanyBySlug,
  isIndexCompany,
  relatedCompanies,
  type SeoSentimentCompany,
} from "@/lib/utils/tickers";

const CACHE_TTL_MS = 24 * 60 * 60_000; // 24h — programmatic pages don't need to be minute-fresh
const CACHE_JITTER_MS = 6 * 60 * 60_000; // spread expiries over a 6h window
// Used instead of the full TTL when a regeneration attempt reused stale content (blocked by
// the llm7 rate limit or a failed/thin response) — retry again soon rather than waiting a
// full day to get real content back, while the rate limit still caps how often llm7 is hit.
const RETRY_TTL_MS = 15 * 60_000;

type CachedRow = NonNullable<Awaited<ReturnType<typeof sentimentPageCache.getBySlug>>>;

/** Deterministic per-slug offset so the growing company list doesn't all expire in the same
 * instant — a same-minute burst of cache misses (e.g. after a sitemap re-crawl) would blow
 * through the shared llm7-seo-sentiment rate limit and dump extra pages onto fallback text. */
function jitterForSlug(slug: string): number {
  let hash = 0;
  for (let i = 0; i < slug.length; i++) {
    hash = (hash * 31 + slug.charCodeAt(i)) >>> 0;
  }
  return hash % CACHE_JITTER_MS;
}

/** Live index numbers backing a weekly recap page — only set for ^DJI/^IXIC/^GSPC. */
export interface IndexWeeklySnapshot {
  weekOfLabel: string;
  price: number | null;
  dayChangePercent: number | null;
  weekChangePercent: number | null;
}

export interface SeoSentimentPageData {
  company: SeoSentimentCompany;
  sections: SeoPageSections;
  overlay: SentimentPricePoint[];
  indexWeekly: IndexWeeklySnapshot | null;
  heroImageUrl: string | null;
  related: SeoSentimentCompany[];
  /** Rolls forward on every 24h cache refresh — use as Article `dateModified`. */
  generatedAt: string;
  /** Stable since first generation — use as Article `datePublished`. */
  firstGeneratedAt: string;
}

async function buildIndexWeeklySnapshot(ticker: string): Promise<IndexWeeklySnapshot> {
  const snapshots = await getMarketIndexSnapshots();
  const snapshot = snapshots.find((s) => s.symbol === ticker) ?? null;
  return {
    weekOfLabel: formatWeekOfLabel(),
    price: snapshot?.price ?? null,
    dayChangePercent: snapshot?.changePercent ?? null,
    weekChangePercent: snapshot?.weekChangePercent ?? null,
  };
}

/** `staleCached`, when given, is the just-expired cache row for this slug — its content is
 * preferred over the generic template whenever regeneration is skipped, rate-limited, or fails,
 * rather than replacing real (if slightly stale) copy with boilerplate. */
async function generateFresh(
  company: SeoSentimentCompany,
  staleCached: CachedRow | null
): Promise<SeoSentimentPageData> {
  const overlay = await getSentimentPriceOverlay(company.ticker);
  const staleSections: SeoPageSections | null = staleCached
    ? JSON.parse(staleCached.sectionsJson)
    : null;

  let indexWeekly: IndexWeeklySnapshot | null = null;
  let sections: SeoPageSections;
  let isFreshGeneration: boolean;

  if (isIndexCompany(company)) {
    indexWeekly = await buildIndexWeeklySnapshot(company.ticker);
    ({ sections, isFreshGeneration } = await generateIndexWeeklyRecapSections(
      {
        companyName: company.companyName,
        ticker: company.ticker,
        weekOfLabel: indexWeekly.weekOfLabel,
        price: indexWeekly.price,
        dayChangePercent: indexWeekly.dayChangePercent,
        weekChangePercent: indexWeekly.weekChangePercent,
        overlay,
      },
      staleSections
    ));
  } else {
    ({ sections, isFreshGeneration } = await generateSeoPageSections(
      company.companyName,
      company.ticker,
      overlay,
      staleSections
    ));
  }

  const hero = await getOrFetchUnsplashImage(
    hashCacheKey(`seo:${company.ticker}`),
    `${company.companyName} stock market`
  );

  const slug = companySlug(company.companyName);
  const expiresAt = new Date(
    Date.now() + (isFreshGeneration ? CACHE_TTL_MS + jitterForSlug(slug) : RETRY_TTL_MS)
  ).toISOString();

  const row = sentimentPageCache.upsert({
    slug,
    ticker: company.ticker,
    sectionsJson: JSON.stringify(sections),
    priceJson: JSON.stringify(overlay),
    // Repurposes the otherwise-unused sentimentJson column to persist the index weekly
    // snapshot, so a cache hit doesn't need to re-fetch it.
    sentimentJson: indexWeekly ? JSON.stringify(indexWeekly) : null,
    expiresAt,
    // Preserve the old generatedAt when reusing stale content so dateModified keeps
    // reflecting the last time the copy actually changed, not the last retry attempt.
    generatedAt: isFreshGeneration ? undefined : staleCached?.generatedAt,
  });

  return {
    company,
    sections,
    overlay,
    indexWeekly,
    heroImageUrl: hero?.imageUrl ?? null,
    related: relatedCompanies(company),
    generatedAt: row.generatedAt,
    firstGeneratedAt: row.firstGeneratedAt ?? row.generatedAt,
  };
}

// Module-scope (process-wide) in-flight dedup, keyed by slug. React's `cache()`
// below only dedupes calls within a single request's render (generateMetadata +
// the page component); it can't stop two *different* requests (e.g. two
// visitors, or a crawler + a visitor) from both missing the DB cache and both
// firing `generateFresh` — which hits the paid llm7 API — at the same time.
const inFlightGenerations = new Map<string, Promise<SeoSentimentPageData>>();

function generateFreshDeduped(company: SeoSentimentCompany, slug: string, staleCached: CachedRow | null) {
  const pending = inFlightGenerations.get(slug);
  if (pending) return pending;

  const promise = generateFresh(company, staleCached).finally(() => {
    inFlightGenerations.delete(slug);
  });
  inFlightGenerations.set(slug, promise);
  return promise;
}

/**
 * Loads (or regenerates, TTL-refreshed) a programmatic SEO sentiment page's data.
 * Wrapped in React's `cache()` so `generateMetadata` and the page component — which both
 * call this for the same request — share one in-flight call instead of each triggering their
 * own `generateFresh`, which was double-hitting the llm7 API per page load on a cache miss.
 */
export const getSeoSentimentPageData = cache(async function getSeoSentimentPageData(
  slug: string
): Promise<SeoSentimentPageData | null> {
  const company = findSeoCompanyBySlug(slug);
  if (!company) return null;

  const cached = await sentimentPageCache.getBySlug(slug);
  if (cached && sentimentPageCache.isFresh(cached)) {
    // DB-cache-backed (see getOrFetchUnsplashImage), so this is a cheap local
    // lookup on a cache hit, not a re-fetch from Unsplash.
    const hero = await getOrFetchUnsplashImage(
      hashCacheKey(`seo:${company.ticker}`),
      `${company.companyName} stock market`
    );

    return {
      company,
      sections: JSON.parse(cached.sectionsJson),
      overlay: cached.priceJson ? JSON.parse(cached.priceJson) : [],
      indexWeekly: cached.sentimentJson ? JSON.parse(cached.sentimentJson) : null,
      heroImageUrl: hero?.imageUrl ?? null,
      related: relatedCompanies(company),
      generatedAt: cached.generatedAt,
      firstGeneratedAt: cached.firstGeneratedAt ?? cached.generatedAt,
    };
  }

  return generateFreshDeduped(company, slug, cached ?? null);
});

export function getAllSeoSentimentSlugs(): string[] {
  return SEO_SENTIMENT_COMPANIES.map((c) => companySlug(c.companyName));
}
