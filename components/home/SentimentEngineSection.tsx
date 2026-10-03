import Link from "next/link";
import { Layers, Sparkles, LineChart } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const FEATURES = [
  {
    icon: Layers,
    title: "News, StockTwits & Reddit in one score",
    description:
      "Every stock sentiment score blends AI-scored news coverage with real-time StockTwits bullish/bearish positioning and Reddit chatter pulled via ApeWisdom — so you're not checking three tabs to read AAPL, TSLA, or NVDA sentiment.",
    detail: "Multi-source aggregation",
  },
  {
    icon: Sparkles,
    title: "Plain-language sentiment analysis",
    description:
      "Each news article is read and scored bullish, bearish, or neutral by AI — then summarized into a plain-English answer to \"why did this stock move,\" not just another sentiment number.",
    detail: "AI-powered, per article",
  },
  {
    icon: LineChart,
    title: "Sentiment vs. price, side by side",
    description:
      "A 90-day rolling overlay chart plots daily news sentiment against price on every stock page, so you can see whether the crowd's mood led the move or lagged it — for any ticker, not just the Magnificent Seven.",
    detail: "90-day rolling overlay",
  },
];

export function SentimentEngineSection() {
  return (
    <section
      aria-labelledby="sentiment-engine-heading"
      className="relative z-10 w-full max-w-5xl px-4 py-16"
    >
      <div className="mx-auto max-w-2xl text-center">
        <p className="text-xs font-medium tracking-widest text-primary uppercase">
          How the sentiment score works
        </p>
        <h2
          id="sentiment-engine-heading"
          className="mt-3 font-serif text-3xl font-medium tracking-tight text-balance sm:text-4xl"
        >
          AI stock sentiment analysis, built for a quick read
        </h2>
        <p className="mt-3 text-muted-foreground">
          Every ticker&rsquo;s sentiment score comes from real news, social, and market data —
          scored by AI and explained in plain language, not buried in a raw feed.
        </p>
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, description, detail }) => (
          <Card key={title} className="text-left">
            <CardContent className="flex flex-col gap-3">
              <div className="flex size-9 items-center justify-center rounded-none bg-primary/10 text-primary">
                <Icon className="size-4.5" aria-hidden="true" />
              </div>
              <h3 className="font-medium">{title}</h3>
              <p className="text-sm text-muted-foreground">{description}</p>
              <p className="mt-auto text-xs text-primary">{detail}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <p className="mt-6 text-center text-sm text-muted-foreground">
        <Link href="/blog/how-we-classify-news-sentiment" className="text-foreground underline">
          Read the full sentiment methodology
        </Link>{" "}
        for how each article is scored, or see{" "}
        <Link href="/about" className="text-foreground underline">
          what Stock Sentiment is
        </Link>
        .
      </p>
    </section>
  );
}
