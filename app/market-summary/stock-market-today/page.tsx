import type { Metadata } from "next";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import { MarketSummaryArticle } from "@/components/marketSummary/MarketSummaryArticle";

const title = "Stock Market Today — Latest Wrap";
const description =
  "What's happening in the stock market today — always the latest AI-generated wrap.";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: "/market-summary/stock-market-today" },
  openGraph: {
    title,
    description,
    url: "/market-summary/stock-market-today",
    type: "website",
  },
  twitter: { card: "summary_large_image", title, description },
};

export const dynamic = "force-dynamic";

export default async function StockMarketTodayPage() {
  const latest = await marketSummary.getLatest();

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      {latest ? (
        <MarketSummaryArticle
          title={latest.title}
          body={latest.body}
          indexSnapshotJson={latest.indexSnapshotJson}
          createdAt={latest.createdAt}
          slug={latest.slug}
          imageUrl={latest.imageUrl}
          imagePhotographerName={latest.imagePhotographerName}
          imagePhotographerProfileUrl={latest.imagePhotographerProfileUrl}
        />
      ) : (
        <p className="text-sm text-muted-foreground">No market summary available yet.</p>
      )}
    </main>
  );
}
