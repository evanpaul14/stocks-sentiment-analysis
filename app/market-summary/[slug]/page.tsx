import type { Metadata } from "next";
import { notFound } from "next/navigation";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import { MarketSummaryArticle } from "@/components/marketSummary/MarketSummaryArticle";

interface MarketSummarySlugPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({
  params,
}: MarketSummarySlugPageProps): Promise<Metadata> {
  const { slug } = await params;
  const record = await marketSummary.getBySlug(slug);
  if (!record) return {};
  const description = record.body.slice(0, 160);
  const canonical = `/market-summary/${slug}`;
  return {
    title: record.title,
    description,
    alternates: { canonical },
    openGraph: {
      title: record.title,
      description,
      url: canonical,
      type: "article",
      publishedTime: record.createdAt,
      ...(record.imageUrl ? { images: [record.imageUrl] } : {}),
    },
    twitter: {
      card: "summary_large_image",
      title: record.title,
      description,
    },
  };
}

export const dynamic = "force-dynamic";

export default async function MarketSummarySlugPage({ params }: MarketSummarySlugPageProps) {
  const { slug } = await params;
  const record = await marketSummary.getBySlug(slug);
  if (!record) notFound();

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <MarketSummaryArticle
        title={record.title}
        body={record.body}
        indexSnapshotJson={record.indexSnapshotJson}
        createdAt={record.createdAt}
        slug={record.slug}
        imageUrl={record.imageUrl}
        imagePhotographerName={record.imagePhotographerName}
        imagePhotographerProfileUrl={record.imagePhotographerProfileUrl}
      />
    </main>
  );
}
