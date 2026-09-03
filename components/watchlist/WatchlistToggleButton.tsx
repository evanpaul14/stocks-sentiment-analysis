"use client";

import { Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWatchlist } from "@/lib/watchlist/useWatchlist";

interface WatchlistToggleButtonProps {
  symbol: string;
  companyName: string;
  price: number | null;
  changePercent: number | null;
}

export function WatchlistToggleButton({
  symbol,
  companyName,
  price,
  changePercent,
}: WatchlistToggleButtonProps) {
  const { add, remove, has } = useWatchlist();
  const inWatchlist = has(symbol);

  return (
    <Button
      type="button"
      variant={inWatchlist ? "secondary" : "outline"}
      size="sm"
      data-umami-event={inWatchlist ? "watchlist-remove" : "watchlist-add"}
      onClick={() => {
        if (inWatchlist) {
          remove(symbol);
        } else {
          add({
            symbol,
            companyName,
            lastPrice: price,
            lastChangePercent: changePercent,
          });
        }
      }}
    >
      <Star
        className={inWatchlist ? "fill-current" : ""}
        data-icon="inline-start"
      />
      {inWatchlist ? "In Watchlist" : "Add to Watchlist"}
    </Button>
  );
}
