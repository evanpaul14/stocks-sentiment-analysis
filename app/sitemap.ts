import type { MetadataRoute } from "next";
import { getAllBlogPosts } from "@/lib/blog/posts";
import { getAllSeoSentimentSlugs } from "@/lib/blog/seoSentimentPageData";
import { SEO_SENTIMENT_COMPANIES } from "@/lib/utils/tickers";
import { toIsoDateTime } from "@/lib/utils/dates";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import * as sentimentPageCache from "@/lib/db/queries/sentimentPageCache";

export const dynamic = "force-dynamic";

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const baseUrl = process.env.SITE_BASE_URL ?? "http://localhost:3000";
  const now = new Date();

  const staticRoutes: MetadataRoute.Sitemap = [
    { url: `${baseUrl}/`, changeFrequency: "daily", priority: 1, lastModified: now },
    { url: `${baseUrl}/trending`, changeFrequency: "hourly", priority: 0.7, lastModified: now },
    {
      url: `${baseUrl}/trending/stocktwits`,
      changeFrequency: "hourly",
      priority: 0.6,
      lastModified: now,
    },
    {
      url: `${baseUrl}/trending/reddit`,
      changeFrequency: "hourly",
      priority: 0.6,
      lastModified: now,
    },
    {
      url: `${baseUrl}/trending/volume`,
      changeFrequency: "hourly",
      priority: 0.6,
      lastModified: now,
    },
    {
      url: `${baseUrl}/market-summary`,
      changeFrequency: "daily",
      priority: 0.8,
      lastModified: now,
    },
    {
      url: `${baseUrl}/market-summary/stock-market-today`,
      changeFrequency: "daily",
      priority: 0.8,
      lastModified: now,
    },
    { url: `${baseUrl}/blog`, changeFrequency: "weekly", priority: 0.6 },
    { url: `${baseUrl}/about`, changeFrequency: "yearly", priority: 0.3 },
    { url: `${baseUrl}/contact`, changeFrequency: "yearly", priority: 0.2 },
    { url: `${baseUrl}/privacy`, changeFrequency: "yearly", priority: 0.1 },
  ];

  // publishedAt is the best available proxy for a static MDX post's last
  // modification — these are hand-edited rarely, so there's no better signal.
  const blogRoutes: MetadataRoute.Sitemap = getAllBlogPosts().map((post) => ({
    url: `${baseUrl}/blog/${post.slug}`,
    changeFrequency: "monthly",
    priority: 0.5,
    lastModified: new Date(post.frontmatter.publishedAt),
  }));

  let generatedAtBySlug = new Map<string, string>();
  try {
    const rows = await sentimentPageCache.listSlugsAndGeneratedAt();
    generatedAtBySlug = new Map(rows.map((row) => [row.slug, row.generatedAt]));
  } catch {
    // DB not ready yet (e.g. very first build) — falls back to `now` below
  }

  const seoSentimentRoutes: MetadataRoute.Sitemap = getAllSeoSentimentSlugs().map((slug) => {
    const generatedAt = generatedAtBySlug.get(slug);
    return {
      url: `${baseUrl}/blog/${slug}`,
      changeFrequency: "daily",
      priority: 0.6,
      lastModified: generatedAt ? new Date(toIsoDateTime(generatedAt)) : now,
    };
  });

  // Price/sentiment data refreshes live, matching the "hourly" changeFrequency.
  const stockRoutes: MetadataRoute.Sitemap = SEO_SENTIMENT_COMPANIES.filter(
    (company) => !company.groups.includes("index")
  ).map((company) => ({
    url: `${baseUrl}/stock/${company.ticker}`,
    changeFrequency: "hourly",
    priority: 0.7,
    lastModified: now,
  }));

  let marketSummaryRoutes: MetadataRoute.Sitemap = [];
  try {
    const archive = await marketSummary.listArchive(60);
    marketSummaryRoutes = archive.map((entry) => ({
      url: `${baseUrl}/market-summary/${entry.slug}`,
      changeFrequency: "never",
      priority: 0.4,
      lastModified: new Date(`${entry.date}T00:00:00Z`),
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
