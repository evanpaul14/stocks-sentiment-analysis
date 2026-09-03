import type { Metadata } from "next";
import Link from "next/link";
import * as marketSummary from "@/lib/db/queries/marketWrap";
import { MarketSummaryArticle } from "@/components/marketSummary/MarketSummaryArticle";
import { EmailSubscribeForm } from "@/components/marketSummary/EmailSubscribeForm";

const title = "Market Summary — Daily Wrap";
const description = "The latest AI-generated end-of-day market summary and archive.";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: "/market-summary" },
  openGraph: { title, description, url: "/market-summary", type: "website" },
  twitter: { card: "summary_large_image", title, description },
};

export const dynamic = "force-dynamic";

export default async function MarketSummaryPage() {
  const [latest, archive] = await Promise.all([
    marketSummary.getLatest(),
    marketSummary.listArchive(10),
  ]);

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

      <section className="mt-8 rounded-xl border border-border bg-card p-4">
        <h2 className="mb-2 text-sm font-medium">Get the daily wrap by email</h2>
        <EmailSubscribeForm />
      </section>

      {archive.length > 1 && (
        <section className="mt-10">
          <h2 className="mb-3 text-lg font-medium">Archive</h2>
          <ul className="space-y-1">
            {archive.slice(1).map((entry) => (
              <li key={entry.slug}>
                <Link
                  href={`/market-summary/${entry.slug}`}
                  className="text-sm hover:underline"
                >
                  {entry.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
