import * as sentimentPageCache from "@/lib/db/queries/sentimentPageCache";
import { getSentimentPriceOverlay, type SentimentPricePoint } from "@/lib/sentiment/sentimentPriceOverlay";
import { generateSeoPageSections, type SeoPageSections } from "@/lib/integrations/llm7/seoSentimentPage";
import { getOrFetchUnsplashImage, hashCacheKey } from "@/lib/integrations/unsplash";
import {
  SEO_SENTIMENT_COMPANIES,
  companySlug,
  findSeoCompanyBySlug,
  relatedCompanies,
  type SeoSentimentCompany,
} from "@/lib/utils/tickers";

const CACHE_TTL_MS = 24 * 60 * 60_000; // 24h — programmatic pages don't need to be minute-fresh

export interface SeoSentimentPageData {
  company: SeoSentimentCompany;
  sections: SeoPageSections;
  overlay: SentimentPricePoint[];
  heroImageUrl: string | null;
  related: SeoSentimentCompany[];
  generatedAt: string;
}

async function generateFresh(company: SeoSentimentCompany): Promise<SeoSentimentPageData> {
  const overlay = await getSentimentPriceOverlay(company.ticker);
  const sections = await generateSeoPageSections(company.companyName, company.ticker, overlay);
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
    sentimentJson: null,
    expiresAt: new Date(Date.now() + CACHE_TTL_MS).toISOString(),
  });

  return {
    company,
    sections,
    overlay,
    heroImageUrl: hero?.imageUrl ?? null,
    related: relatedCompanies(company),
    generatedAt: row.generatedAt,
  };
}

/** Loads (or regenerates, TTL-refreshed) a programmatic SEO sentiment page's data. */
export async function getSeoSentimentPageData(slug: string): Promise<SeoSentimentPageData | null> {
  const company = findSeoCompanyBySlug(slug);
  if (!company) return null;

  const cached = await sentimentPageCache.getBySlug(slug);
  if (cached && sentimentPageCache.isFresh(cached)) {
    return {
      company,
      sections: JSON.parse(cached.sectionsJson),
      overlay: cached.priceJson ? JSON.parse(cached.priceJson) : [],
      heroImageUrl: null,
      related: relatedCompanies(company),
      generatedAt: cached.generatedAt,
    };
  }

  return generateFresh(company);
}

export function getAllSeoSentimentSlugs(): string[] {
  return SEO_SENTIMENT_COMPANIES.map((c) => companySlug(c.companyName));
}
