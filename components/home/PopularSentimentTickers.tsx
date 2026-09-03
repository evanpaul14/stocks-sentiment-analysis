"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { SEO_SENTIMENT_COMPANIES, companySlug } from "@/lib/utils/tickers";

interface PriceInfo {
  price: number | null;
  changePercent: number | null;
}

export function PopularSentimentTickers() {
  const containerRef = useRef<HTMLDivElement>(null);
  const setRef = useRef<HTMLDivElement>(null);
  // Number of times SEO_SENTIMENT_COMPANIES is repeated to form one "half" of
  // the track. The track renders two of these halves back to back and loops
  // by translateX(-50%), so each half must be at least as wide as the
  // viewport or the loop shows a blank gap before it repeats.
  const [repeat, setRepeat] = useState(1);
  const [prices, setPrices] = useState<Record<string, PriceInfo>>({});

  useEffect(() => {
    let cancelled = false;
    async function loadPrices() {
      try {
        const response = await fetch("/api/trending/prices", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbols: SEO_SENTIMENT_COMPANIES.map((c) => c.ticker),
          }),
        });
        if (!response.ok || cancelled) return;
        const snapshots: Array<{
          symbol: string;
          price: number | null;
          changePercent: number | null;
        }> = await response.json();
        if (cancelled) return;
        setPrices(
          Object.fromEntries(
            snapshots.map((s) => [s.symbol, { price: s.price, changePercent: s.changePercent }])
          )
        );
      } catch {
        // silent: fall back to ticker-only display
      }
    }
    loadPrices();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    const set = setRef.current;
    if (!container || !set) return;

    const measure = () => {
      const setWidth = set.scrollWidth / repeat;
      const containerWidth = container.clientWidth;
      if (setWidth <= 0) return;
      const needed = Math.ceil(containerWidth / setWidth) + 1;
      setRepeat((current) => (needed > current ? needed : current));
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(container);
    return () => observer.disconnect();
  }, [repeat]);

  const baseSet = Array.from({ length: repeat }, () => SEO_SENTIMENT_COMPANIES).flat();
  const items = [...baseSet, ...baseSet];
  // Keep scroll speed constant (~px/s of the single-copy baseline) as repeat grows.
  const durationSeconds = 28 * repeat;

  return (
    <div ref={containerRef} className="relative z-10 w-full overflow-hidden border-y border-border bg-card/60 backdrop-blur">
      <div
        ref={setRef}
        className="flex w-max gap-8 py-2 whitespace-nowrap [animation-name:ticker-tape] [animation-timing-function:linear] [animation-iteration-count:infinite] hover:[animation-play-state:paused]"
        style={{ animationDuration: `${durationSeconds}s` }}
      >
        {items.map((company, index) => {
          const info = prices[company.ticker];
          const isUp = (info?.changePercent ?? 0) >= 0;
          return (
            <Link
              key={`${company.ticker}-${index}`}
              href={`/blog/${companySlug(company.companyName)}`}
              className="text-xs text-muted-foreground transition-colors hover:text-foreground"
            >
              <span className="font-medium text-foreground">{company.ticker}</span>{" "}
              {info?.price != null ? (
                <>
                  <span className="tabular-nums">${info.price.toFixed(2)}</span>{" "}
                  {info.changePercent != null && (
                    <span
                      className={`tabular-nums ${isUp ? "text-[var(--color-chart-1)]" : "text-destructive"}`}
                    >
                      {isUp ? "+" : ""}
                      {info.changePercent.toFixed(2)}%
                    </span>
                  )}
                </>
              ) : (
                <span>{company.companyName} sentiment</span>
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
