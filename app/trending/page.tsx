import type { Metadata } from "next";
import { getAllTrendingSourceData } from "@/lib/trending/getTrendingSourceData";
import { TrendingTabs } from "@/components/trending/TrendingTabs";
import { TrendingList } from "@/components/trending/TrendingList";

const title = "Trending Stocks — StockTwits, Reddit & Volume";
const description =
  "See which stocks are trending right now on StockTwits, Reddit, and by trading volume.";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: "/trending" },
  openGraph: { title, description, url: "/trending", type: "website" },
  twitter: { card: "summary_large_image", title, description },
};

export const dynamic = "force-dynamic";

export default async function TrendingPage() {
  const { stocktwits, reddit, volume } = await getAllTrendingSourceData();

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-2 text-2xl font-semibold">Trending</h1>
      <TrendingTabs />

      <div className="space-y-10">
        <section>
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">StockTwits</h2>
          <TrendingList items={stocktwits} />
        </section>
        <section>
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">Reddit</h2>
          <TrendingList items={reddit} />
        </section>
        <section>
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">Volume</h2>
          <TrendingList items={volume} />
        </section>
      </div>
    </main>
  );
}
