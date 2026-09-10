import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "About",
  description: "What Stock Sentiment is and how its sentiment analysis works.",
  alternates: { canonical: "/about" },
};

export default function AboutPage() {
  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">About Stock Sentiment</h1>

      <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          Stock Sentiment tracks real-time prices, historical charts, and news sentiment for
          publicly traded companies and major indices. For each ticker, recent news headlines are
          run through an AI model that classifies them as positive, negative, or neutral, and the
          results are aggregated into a single view alongside live price data.
        </p>
        <p>
          The site is built and maintained by a single developer as an independent project — it
          is not affiliated with any brokerage, exchange, or financial institution. See{" "}
          <Link href="/blog/how-we-classify-news-sentiment" className="text-foreground underline">
            how sentiment is classified
          </Link>{" "}
          for methodology details.
        </p>
        <p>
          Stock Sentiment provides automated, AI-generated market data and sentiment analysis for
          informational purposes only — see the{" "}
          <Link href="/privacy" className="text-foreground underline">
            privacy policy
          </Link>{" "}
          for details on data sources and handling, or{" "}
          <Link href="/contact" className="text-foreground underline">
            get in touch
          </Link>{" "}
          with questions or corrections.
        </p>
      </div>
    </main>
  );
}
