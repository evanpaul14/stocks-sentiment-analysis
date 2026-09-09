"use client";

import { useState } from "react";
import Image from "next/image";

interface CompanyLogoProps {
  symbol: string;
  companyName: string;
  size?: number;
}

/**
 * StockTwits serves ticker-keyed logo images (same CDN the old app's cached
 * StockTwits responses reference, e.g. logos.stocktwits-cdn.com/CRM.png) —
 * no API call needed, just the symbol. Not every ticker has one, so this
 * falls back to a letter avatar on load error.
 */
export function CompanyLogo({ symbol, companyName, size = 40 }: CompanyLogoProps) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div
        className="flex shrink-0 items-center justify-center rounded-lg border border-border bg-muted text-sm font-medium text-muted-foreground"
        style={{ width: size, height: size }}
        aria-hidden="true"
      >
        {companyName.charAt(0).toUpperCase()}
      </div>
    );
  }

  return (
    <Image
      src={`https://logos.stocktwits-cdn.com/${encodeURIComponent(symbol.toUpperCase())}.png`}
      alt=""
      width={size}
      height={size}
      className="shrink-0 rounded-lg border border-border bg-card object-contain"
      style={{ width: size, height: size }}
      onError={() => setFailed(true)}
    />
  );
}
