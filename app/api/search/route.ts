import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { resolveSymbol, SymbolNotFoundError } from "@/lib/integrations/yahoo/search";
import { getStockInfo } from "@/lib/integrations/yahoo/quote";
import { getHistoricalPrices } from "@/lib/integrations/yahoo/historical";
import { getNewsArticles } from "@/lib/integrations/news/googleNews";

/**
 * Resolves a free-text query, then returns stock_info + historical_data +
 * articles. Sentiment is deliberately excluded here to keep this fast — the
 * frontend streams per-article sentiment via /api/sentiment after render.
 */
async function handler(request: NextRequest) {
  const body = await request.json().catch(() => null);
  const query = typeof body?.query === "string" ? body.query.trim() : "";

  if (!query) {
    return NextResponse.json({ error: "Missing query" }, { status: 400 });
  }

  try {
    const symbol = await resolveSymbol(query);

    const [stockInfo, historicalData, articles] = await Promise.all([
      getStockInfo(symbol),
      getHistoricalPrices(symbol, "1d"),
      getNewsArticles(symbol, 10),
    ]);

    return NextResponse.json({
      stock_info: stockInfo,
      historical_data: historicalData,
      articles,
    });
  } catch (error) {
    if (error instanceof SymbolNotFoundError) {
      return NextResponse.json({ error: "Company not found" }, { status: 404 });
    }
    console.error("[api/search] failed", error);
    return NextResponse.json({ error: "Search failed" }, { status: 502 });
  }
}

export const POST = withRateLimit(
  { routeName: "search", limit: 10, windowMs: 60_000 },
  handler
);
