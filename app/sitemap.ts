import type { MetadataRoute } from "next";
import { getAllBlogSlugs } from "@/lib/blog/posts";
import { getAllSeoSentimentSlugs } from "@/lib/blog/seoSentimentPageData";
import { SEO_SENTIMENT_COMPANIES } from "@/lib/utils/tickers";
import * as marketSummary from "@/lib/db/queries/marketWrap";

export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = process.env.SITE_BASE_URL ?? "http://localhost:3000";

  const staticRoutes: MetadataRoute.Sitemap = [
    { url: `${baseUrl}/`, changeFrequency: "daily", priority: 1 },
    { url: `${baseUrl}/trending`, changeFrequency: "hourly", priority: 0.7 },
    { url: `${baseUrl}/trending/stocktwits`, changeFrequency: "hourly", priority: 0.6 },
    { url: `${baseUrl}/trending/reddit`, changeFrequency: "hourly", priority: 0.6 },
    { url: `${baseUrl}/trending/volume`, changeFrequency: "hourly", priority: 0.6 },
    { url: `${baseUrl}/market-summary`, changeFrequency: "daily", priority: 0.8 },
    {
      url: `${baseUrl}/market-summary/stock-market-today`,
      changeFrequency: "daily",
      priority: 0.8,
    },
    { url: `${baseUrl}/blog`, changeFrequency: "weekly", priority: 0.6 },
    { url: `${baseUrl}/privacy`, changeFrequency: "yearly", priority: 0.1 },
  ];

  const blogRoutes: MetadataRoute.Sitemap = getAllBlogSlugs().map((slug) => ({
    url: `${baseUrl}/blog/${slug}`,
    changeFrequency: "monthly",
    priority: 0.5,
  }));

  const seoSentimentRoutes: MetadataRoute.Sitemap = getAllSeoSentimentSlugs().map((slug) => ({
    url: `${baseUrl}/blog/${slug}`,
    changeFrequency: "daily",
    priority: 0.6,
  }));

  const stockRoutes: MetadataRoute.Sitemap = SEO_SENTIMENT_COMPANIES.filter(
    (company) => !company.groups.includes("index")
  ).map((company) => ({
    url: `${baseUrl}/stock/${company.ticker}`,
    changeFrequency: "hourly",
    priority: 0.7,
  }));

  let marketSummaryRoutes: MetadataRoute.Sitemap = [];
  try {
    const archive = await marketSummary.listArchive(60);
    marketSummaryRoutes = archive.map((entry) => ({
      url: `${baseUrl}/market-summary/${entry.slug}`,
      changeFrequency: "never",
      priority: 0.4,
    }));
  } catch {
    // DB not ready yet (e.g. very first build) — sitemap still returns static routes
  }

  return [
    ...staticRoutes,
    ...blogRoutes,
    ...seoSentimentRoutes,
    ...stockRoutes,
    ...marketSummaryRoutes,
  ];
}
