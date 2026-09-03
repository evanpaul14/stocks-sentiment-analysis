import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  getTrendingSourceData,
  type TrendingSource,
} from "@/lib/trending/getTrendingSourceData";
import { TrendingTabs } from "@/components/trending/TrendingTabs";
import { TrendingList } from "@/components/trending/TrendingList";

const VALID_SOURCES: TrendingSource[] = ["stocktwits", "reddit", "volume"];
const LABELS: Record<TrendingSource, string> = {
  stocktwits: "StockTwits",
  reddit: "Reddit",
  volume: "Volume",
};

interface TrendingSourcePageProps {
  params: Promise<{ source: string }>;
}

export async function generateMetadata({
  params,
}: TrendingSourcePageProps): Promise<Metadata> {
  const { source } = await params;
  if (!VALID_SOURCES.includes(source as TrendingSource)) return {};
  const label = LABELS[source as TrendingSource];
  const title = `Trending on ${label} — Stock Sentiment`;
  const description = `Stocks trending right now on ${label}.`;
  const canonical = `/trending/${source}`;
  return {
    title,
    description,
    alternates: { canonical },
    openGraph: { title, description, url: canonical, type: "website" },
    twitter: { card: "summary_large_image", title, description },
  };
}

export const dynamic = "force-dynamic";

export default async function TrendingSourcePage({ params }: TrendingSourcePageProps) {
  const { source } = await params;
  if (!VALID_SOURCES.includes(source as TrendingSource)) notFound();

  const typedSource = source as TrendingSource;
  const items = await getTrendingSourceData(typedSource);

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-2 text-2xl font-semibold">Trending</h1>
      <TrendingTabs active={typedSource} />
      <TrendingList items={items} />
    </main>
  );
}
