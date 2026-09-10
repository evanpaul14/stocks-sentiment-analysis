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
  generatedAt: string;
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

async function generateFresh(company: SeoSentimentCompany): Promise<SeoSentimentPageData> {
  const overlay = await getSentimentPriceOverlay(company.ticker);

  let indexWeekly: IndexWeeklySnapshot | null = null;
  let sections: SeoPageSections;

  if (isIndexCompany(company)) {
    indexWeekly = await buildIndexWeeklySnapshot(company.ticker);
    sections = await generateIndexWeeklyRecapSections({
      companyName: company.companyName,
      ticker: company.ticker,
      weekOfLabel: indexWeekly.weekOfLabel,
      price: indexWeekly.price,
      dayChangePercent: indexWeekly.dayChangePercent,
      weekChangePercent: indexWeekly.weekChangePercent,
      overlay,
    });
  } else {
    sections = await generateSeoPageSections(company.companyName, company.ticker, overlay);
  }

  const hero = await getOrFetchUnsplashImage(
    hashCacheKey(`seo:${company.ticker}`),
    `${company.companyName} stock market`
  );

  const slug = companySlug(company.companyName);
  const row = sentimentPageCache.upsert({
    slug,
    ticker: company.ticker,
    sectionsJson: JSON.stringify(sections),
    priceJson: JSON.stringify(overlay),
    // Repurposes the otherwise-unused sentimentJson column to persist the index weekly
    // snapshot, so a cache hit doesn't need to re-fetch it.
    sentimentJson: indexWeekly ? JSON.stringify(indexWeekly) : null,
    expiresAt: new Date(Date.now() + CACHE_TTL_MS).toISOString(),
  });

  return {
    company,
    sections,
    overlay,
    indexWeekly,
    heroImageUrl: hero?.imageUrl ?? null,
    related: relatedCompanies(company),
    generatedAt: row.generatedAt,
  };
}

// Module-scope (process-wide) in-flight dedup, keyed by slug. React's `cache()`
// below only dedupes calls within a single request's render (generateMetadata +
// the page component); it can't stop two *different* requests (e.g. two
// visitors, or a crawler + a visitor) from both missing the DB cache and both
// firing `generateFresh` — which hits the paid llm7 API — at the same time.
const inFlightGenerations = new Map<string, Promise<SeoSentimentPageData>>();

function generateFreshDeduped(company: SeoSentimentCompany, slug: string) {
  const pending = inFlightGenerations.get(slug);
  if (pending) return pending;

  const promise = generateFresh(company).finally(() => {
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
    return {
      company,
      sections: JSON.parse(cached.sectionsJson),
      overlay: cached.priceJson ? JSON.parse(cached.priceJson) : [],
      indexWeekly: cached.sentimentJson ? JSON.parse(cached.sentimentJson) : null,
      heroImageUrl: null,
      related: relatedCompanies(company),
      generatedAt: cached.generatedAt,
    };
  }

  return generateFreshDeduped(company, slug);
});

export function getAllSeoSentimentSlugs(): string[] {
  return SEO_SENTIMENT_COMPANIES.map((c) => companySlug(c.companyName));
}
