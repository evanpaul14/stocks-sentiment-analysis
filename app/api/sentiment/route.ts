import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { classifySentiment, ModelOverloadedError } from "@/lib/integrations/sentiment/classify";
import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";
import { getNewsArticles } from "@/lib/integrations/news";
import { getStockInfo } from "@/lib/integrations/yahoo/quote";
import { isPlausibleSymbol } from "@/lib/utils/tickers";

/**
 * Classifies one article from the stock page's news list. The client only
 * names the article (ticker + link); its title/description are taken from
 * the server's own news fetch for that ticker. Results are cached by
 * (ticker, link) and feed the verdict badge and sentiment history for every
 * visitor, so accepting client-supplied article text would let anyone
 * pre-seed a misleading label for a real article or insert fake history rows.
 */
async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);

  const ticker = typeof body?.ticker === "string" ? body.ticker.trim().toUpperCase() : "";
  const link = typeof body?.link === "string" ? body.link.trim() : "";

  if (!isPlausibleSymbol(ticker) || !link) {
    return NextResponse.json({ error: "Missing ticker or link" }, { status: 400 });
  }

  const existing = await sentimentHistory.findExisting(ticker, link);
  if (existing) {
    return NextResponse.json({ sentiment: existing.sentiment, cached: true });
  }

  try {
    // Same call (and cache key) the stock page used to build its article list.
    const articles = await getNewsArticles(ticker, 10);
    const article = articles.find((a) => a.link === link);
    if (!article) {
      return NextResponse.json({ error: "Unknown article" }, { status: 404 });
    }

    const { companyName } = await getStockInfo(ticker);
    const sentiment = await classifySentiment(companyName, article.title, article.description ?? "");

    const row = await sentimentHistory.insert({
      ticker,
      articleTitle: article.title,
      articleLink: article.link,
      articleSource: article.source ?? null,
      articlePublishedAt: article.publishedAt ?? null,
      sentiment,
    });

    // A concurrent request may have inserted this (ticker, link) first; serve
    // its result instead of erroring, since classification still succeeded.
    return NextResponse.json({ sentiment: row?.sentiment ?? sentiment, cached: false });
  } catch (error) {
    if (error instanceof ModelOverloadedError) {
      return NextResponse.json({ error: "MODEL_OVERLOADED" }, { status: 503 });
    }
    console.error("[api/sentiment] classification failed", error);
    return NextResponse.json({ error: "Sentiment analysis failed" }, { status: 502 });
  }
}

export const POST = withRateLimit(
  { routeName: "sentiment", limit: 60, windowMs: 60_000 },
  handler
);
