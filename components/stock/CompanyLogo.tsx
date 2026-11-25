"use client";

import { useState } from "react";
import Image from "next/image";

interface CompanyLogoProps {
  symbol: string;
  companyName: string;
  size?: number;
}

/**
 * The CDN resizes via `?w=`, but only for a fixed allowlist of widths — any
 * other value (e.g. 100, 200) returns 403, which would trip the fallback avatar.
 * Originals are 1000x1000.
 */
const CDN_WIDTHS = [16, 20, 24, 32, 40, 48, 60, 64, 72, 80, 96, 128, 192] as const;

/**
 * Smallest allowed CDN width covering a 4x render, capped at the largest.
 * 4x rather than 2x: the CDN's small webps are heavily compressed, and 3x
 * phone screens plus browser zoom made a 2x source look soft. Even the
 * 192px cap is only ~1.5KB.
 */
function cdnWidth(size: number): number {
  return CDN_WIDTHS.find((w) => w >= size * 4) ?? CDN_WIDTHS[CDN_WIDTHS.length - 1];
}

/**
 * StockTwits serves ticker-keyed logo images (same CDN the old app's cached
 * StockTwits responses reference, e.g. logos.stocktwits-cdn.com/CRM.png; the CDN also serves .webp, used here) —
 * no API call needed, just the symbol. Not every ticker has one, so this
 * falls back to a letter avatar on load error.
 */
export function CompanyLogo({ symbol, companyName, size = 40 }: CompanyLogoProps) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div
        className="flex shrink-0 items-center justify-center rounded-none border border-border bg-muted text-sm font-medium text-muted-foreground"
        style={{ width: size, height: size }}
        aria-hidden="true"
      >
        {companyName.charAt(0).toUpperCase()}
      </div>
    );
  }

  return (
    // alt="" is safe only because every call site renders the company name as
    // adjacent text (see app/stock/[symbol]/page.tsx) — if this is ever used
    // standalone, pass alt={companyName} instead.
    <Image
      src={`https://logos.stocktwits-cdn.com/${encodeURIComponent(symbol.toUpperCase())}.webp?w=${cdnWidth(size)}`}
      alt=""
      width={size}
      height={size}
      // The CDN already resizes; running it through Next's optimizer again
      // re-encoded the small webp at quality 75 and capped the 2x srcset
      // entry at the source width, which is what made logos look blurry.
      unoptimized
      className="shrink-0 rounded-none border border-border bg-card object-contain"
      style={{ width: size, height: size }}
      onError={() => setFailed(true)}
    />
  );
}
