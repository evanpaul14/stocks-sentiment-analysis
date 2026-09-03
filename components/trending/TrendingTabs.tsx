"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { TrendingSource } from "@/lib/trending/getTrendingSourceData";

const TABS: Array<{ source: TrendingSource; label: string }> = [
  { source: "stocktwits", label: "StockTwits" },
  { source: "reddit", label: "Reddit" },
  { source: "volume", label: "Volume" },
];

export function TrendingTabs({ active }: { active?: TrendingSource }) {
  const pathname = usePathname();
  const isAll = pathname === "/trending";

  return (
    <nav className="mb-6 flex gap-2 border-b border-border">
      <TabLink href="/trending" label="All" isActive={isAll} />
      {TABS.map((tab) => (
        <TabLink
          key={tab.source}
          href={`/trending/${tab.source}`}
          label={tab.label}
          isActive={active === tab.source}
        />
      ))}
    </nav>
  );
}

function TabLink({ href, label, isActive }: { href: string; label: string; isActive: boolean }) {
  return (
    <Link
      href={href}
      className={`border-b-2 px-3 py-2 text-sm font-medium ${
        isActive
          ? "border-primary text-foreground"
          : "border-transparent text-muted-foreground hover:text-foreground"
      }`}
    >
      {label}
    </Link>
  );
}
