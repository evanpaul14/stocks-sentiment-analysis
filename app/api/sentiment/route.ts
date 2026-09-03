import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { classifySentiment, ModelOverloadedError } from "@/lib/integrations/sentiment/classify";
import * as sentimentHistory from "@/lib/db/queries/sentimentHistory";

interface SentimentRequestBody {
  ticker?: string;
  companyName?: string;
  article?: {
    title?: string;
    description?: string;
    link?: string;
    source?: string;
    publishedAt?: string;
  };
}

async function handler(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as SentimentRequestBody | null;

  const ticker = body?.ticker?.trim().toUpperCase();
  const companyName = body?.companyName?.trim() || ticker;
  const title = body?.article?.title?.trim();
  const description = body?.article?.description?.trim() ?? "";
  const link = body?.article?.link?.trim() || null;

  if (!ticker || !companyName || !title) {
    return NextResponse.json(
      { error: "Missing ticker, companyName, or article.title" },
      { status: 400 }
    );
  }

  const existing = await sentimentHistory.findExisting(ticker, link);
  if (existing) {
    return NextResponse.json({ sentiment: existing.sentiment, cached: true });
  }

  try {
    const sentiment = await classifySentiment(companyName, title, description);

    sentimentHistory.insert({
      ticker,
      articleTitle: title,
      articleLink: link,
      articleSource: body?.article?.source?.trim() || null,
      articlePublishedAt: body?.article?.publishedAt?.trim() || null,
      sentiment,
    });

    return NextResponse.json({ sentiment, cached: false });
  } catch (error) {
    if (error instanceof ModelOverloadedError) {
      return NextResponse.json({ error: "MODEL_OVERLOADED" }, { status: 503 });
    }
    console.error("[api/sentiment] classification failed", error);
    return NextResponse.json({ error: "Sentiment analysis failed" }, { status: 502 });
  }
}

export const POST = withRateLimit(
  { routeName: "sentiment", limit: 30, windowMs: 60_000 },
  handler
);
