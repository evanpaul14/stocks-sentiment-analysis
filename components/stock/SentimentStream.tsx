"use client";

import { useEffect, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import type { NewsArticle } from "@/lib/integrations/news/googleNews";

type SentimentLabel = "positive" | "negative" | "neutral";

interface ArticleSentiment {
  article: NewsArticle;
  sentiment: SentimentLabel | "pending" | "error";
}

interface SentimentStreamProps {
  ticker: string;
  companyName: string;
  articles: NewsArticle[];
}

const SENTIMENT_COLORS: Record<SentimentLabel, string> = {
  positive: "var(--color-chart-1)",
  negative: "var(--color-destructive)",
  neutral: "var(--color-muted-foreground)",
};

export function SentimentStream({ ticker, companyName, articles }: SentimentStreamProps) {
  const [results, setResults] = useState<ArticleSentiment[]>(
    articles.map((article) => ({ article, sentiment: "pending" }))
  );
  useEffect(() => {
    if (articles.length === 0) return;

    let cancelled = false;

    async function run() {
      for (let i = 0; i < articles.length; i++) {
        if (cancelled) return;
        const article = articles[i];

        try {
          const response = await fetch("/api/sentiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              ticker,
              companyName,
              article: {
                title: article.title,
                description: article.description,
                link: article.link,
                source: article.source,
                publishedAt: article.publishedAt,
              },
            }),
          });

          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const data = await response.json();
          if (cancelled) return;

          setResults((prev) => {
            const next = [...prev];
            next[i] = { article, sentiment: data.sentiment as SentimentLabel };
            return next;
          });
        } catch {
          if (cancelled) return;
          setResults((prev) => {
            const next = [...prev];
            next[i] = { article, sentiment: "error" };
            return next;
          });
        }
      }
    }

    run();
    return () => {
      cancelled = true;
    };
  }, [articles, ticker, companyName]);

  const analyzed = results.filter(
    (r) => r.sentiment !== "pending" && r.sentiment !== "error"
  );
  const progressPercent =
    articles.length === 0 ? 100 : Math.round((analyzed.length / articles.length) * 100);

  const counts: Record<SentimentLabel, number> = {
    positive: 0,
    negative: 0,
    neutral: 0,
  };
  for (const r of analyzed) counts[r.sentiment as SentimentLabel]++;

  const chartData = (Object.keys(counts) as SentimentLabel[])
    .filter((label) => counts[label] > 0)
    .map((label) => ({ name: label, value: counts[label] }));

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="h-24 w-24 shrink-0">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  dataKey="value"
                  innerRadius={28}
                  outerRadius={44}
                  isAnimationActive={false}
                >
                  {chartData.map((entry) => (
                    <Cell key={entry.name} fill={SENTIMENT_COLORS[entry.name]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-full items-center justify-center rounded-full border border-border text-xs text-muted-foreground">
              —
            </div>
          )}
        </div>
        <div className="flex-1 space-y-1">
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Analyzed {analyzed.length} of {articles.length} articles
          </p>
          <div className="flex gap-3 text-xs">
            <span className="text-[var(--color-chart-1)]">Positive {counts.positive}</span>
            <span className="text-destructive">Negative {counts.negative}</span>
            <span className="text-muted-foreground">Neutral {counts.neutral}</span>
          </div>
        </div>
      </div>

      <ul className="space-y-2">
        {results.map(({ article, sentiment }) => (
          <li
            key={article.link}
            className="flex items-start justify-between gap-3 rounded-lg border border-border p-3 text-sm"
          >
            <div className="min-w-0">
              <a
                href={article.link}
                target="_blank"
                rel="noopener noreferrer"
                className="line-clamp-2 font-medium hover:underline"
              >
                {article.title}
              </a>
              <p className="mt-0.5 text-xs text-muted-foreground">{article.source}</p>
            </div>
            <SentimentBadge sentiment={sentiment} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function SentimentBadge({ sentiment }: { sentiment: ArticleSentiment["sentiment"] }) {
  if (sentiment === "pending") {
    return <span className="shrink-0 text-xs text-muted-foreground">…</span>;
  }
  if (sentiment === "error") {
    return <span className="shrink-0 text-xs text-destructive">error</span>;
  }
  return (
    <span
      className="shrink-0 rounded-full px-2 py-0.5 text-xs font-medium capitalize"
      style={{
        color: SENTIMENT_COLORS[sentiment],
        border: `1px solid ${SENTIMENT_COLORS[sentiment]}`,
      }}
    >
      {sentiment}
    </span>
  );
}
