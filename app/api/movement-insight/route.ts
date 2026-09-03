import { NextResponse, type NextRequest } from "next/server";
import { withRateLimit } from "@/lib/ratelimit/withRateLimit";
import { buildMovementInsight } from "@/lib/integrations/llm7/movementInsight";

interface MovementInsightBody {
  symbol?: string;
  companyName?: string;
  changePercent?: number;
}

async function handler(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as MovementInsightBody | null;

  const symbol = body?.symbol?.trim().toUpperCase();
  const companyName = body?.companyName?.trim() || symbol;
  const changePercent = body?.changePercent;

  if (!symbol || !companyName || typeof changePercent !== "number") {
    return NextResponse.json(
      { error: "Missing symbol, companyName, or changePercent" },
      { status: 400 }
    );
  }

  try {
    const insight = await buildMovementInsight(symbol, companyName, changePercent);
    return NextResponse.json({ movement_insight: insight });
  } catch (error) {
    console.error("[api/movement-insight] failed", error);
    return NextResponse.json({ movement_insight: null }, { status: 200 });
  }
}

export const POST = withRateLimit(
  { routeName: "movement-insight", limit: 20, windowMs: 60_000 },
  handler
);
