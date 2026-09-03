import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How Stock Sentiment handles your data.",
  alternates: { canonical: "/privacy" },
};

export default function PrivacyPage() {
  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Privacy Policy</h1>

      <div className="prose prose-invert max-w-none space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          Stock Sentiment does not require an account and does not collect personal information
          to use the core product. Your watchlist and recent search history are stored only in
          your browser&apos;s local storage and are never sent to our servers except as part of
          the requests needed to look up the stocks you search for.
        </p>

        <h2 className="text-base font-medium text-foreground">What we store</h2>
        <p>
          If you subscribe to the daily market summary email, we store your email address with
          our email delivery provider (Mailgun) to send you that digest. You can unsubscribe at
          any time using the link in any email we send.
        </p>

        <h2 className="text-base font-medium text-foreground">Third-party data</h2>
        <p>
          Stock prices, news headlines, and social sentiment shown on this site are sourced from
          third-party providers (including Yahoo Finance, Google News, StockTwits, Reddit via
          ApeWisdom, and Alpaca). We are not responsible for the accuracy of third-party data.
        </p>

        <h2 className="text-base font-medium text-foreground">Analytics</h2>
        <p>
          We may use privacy-friendly, cookie-free analytics to understand aggregate site usage.
          This does not identify individual visitors.
        </p>

        <h2 className="text-base font-medium text-foreground">Contact</h2>
        <p>Questions about this policy can be sent to the site operator.</p>
      </div>
    </main>
  );
}
