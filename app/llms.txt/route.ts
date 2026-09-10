import { NextResponse } from "next/server";
import { getAllBlogSlugs } from "@/lib/blog/posts";

export const dynamic = "force-dynamic";

export async function GET() {
  const baseUrl = process.env.SITE_BASE_URL ?? "http://localhost:3000";
  const editorialSlugs = getAllBlogSlugs();

  const editorialLinks = editorialSlugs
    .map((slug) => `- [${slug.replace(/-/g, " ")}](${baseUrl}/blog/${slug})`)
    .join("\n");

  const body = `# Stock Sentiment

> Real-time stock prices, historical charts, and AI-powered news sentiment analysis for public companies and major market indices.

Stock Sentiment classifies recent news headlines for a ticker as positive, negative, or neutral using an AI model, then aggregates the results alongside live price data. Programmatic per-company and per-index sentiment pages are generated daily; a full, current list of them is in the XML sitemap.

## Docs

${editorialLinks}
- [About](${baseUrl}/about): what this site is and who runs it.
- [Privacy Policy](${baseUrl}/privacy): data sources and handling.

## Key pages

- [Market Summary](${baseUrl}/market-summary): daily market wrap with sentiment context.
- [Trending Stocks](${baseUrl}/trending): most-discussed tickers right now, by StockTwits, Reddit, and trading volume.
- [Sitemap](${baseUrl}/sitemap.xml): full, current list of per-company and per-index sentiment pages.
`;

  return new NextResponse(body, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
