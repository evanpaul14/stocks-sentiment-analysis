import Link from "next/link";
import type { TrendingItem } from "@/lib/trending/getTrendingSourceData";

interface TrendingListProps {
  items: TrendingItem[];
}

export function TrendingList({ items }: TrendingListProps) {
  if (items.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-muted-foreground">
        No trending data available right now.
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {items.map((item, index) => (
        <li key={item.symbol}>
          <Link
            href={`/stock/${item.symbol}`}
            className="flex items-center justify-between gap-3 rounded-lg border border-border p-3 text-sm transition-all duration-150 hover:-translate-y-0.5 hover:border-foreground/20 hover:bg-muted hover:shadow-sm active:translate-y-0"
          >
            <div className="flex min-w-0 items-center gap-3">
              <span className="w-5 shrink-0 text-xs text-muted-foreground">{index + 1}</span>
              <div className="min-w-0">
                <p className="font-medium">{item.symbol}</p>
                <p className="truncate text-xs text-muted-foreground">{item.companyName}</p>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-3">
              <MetaBadge item={item} />
              <PriceCell price={item.price} changePercent={item.changePercent} />
            </div>
          </Link>
        </li>
      ))}
    </ul>
  );
}

function PriceCell({ price, changePercent }: { price: number | null; changePercent: number | null }) {
  const isUp = (changePercent ?? 0) >= 0;
  return (
    <div className="text-right tabular-nums">
      <p>{price != null ? `$${price.toFixed(2)}` : "—"}</p>
      {changePercent != null && (
        <p className={`text-xs ${isUp ? "text-[var(--color-chart-1)]" : "text-destructive"}`}>
          {isUp ? "+" : ""}
          {changePercent.toFixed(2)}%
        </p>
      )}
    </div>
  );
}

function MetaBadge({ item }: { item: TrendingItem }) {
  const { meta } = item;
  if (meta.source === "reddit") {
    return (
      <span className="text-xs text-muted-foreground">
        {meta.mentions} mentions
        {meta.tag.type === "trending" && (
          <span className="ml-1 text-[var(--color-chart-1)]">Trending</span>
        )}
        {meta.tag.type === "up-spots" && (
          <span className="ml-1 text-[var(--color-chart-1)]">Up {meta.tag.spots}</span>
        )}
      </span>
    );
  }
  if (meta.source === "volume") {
    return (
      <span className="text-xs text-muted-foreground">
        {(meta.volume / 1e6).toFixed(1)}M vol
      </span>
    );
  }
  return (
    <span className="text-xs text-muted-foreground">
      {meta.watchlistCount.toLocaleString()} watching
    </span>
  );
}
