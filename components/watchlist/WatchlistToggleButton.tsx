"use client";

import { useState } from "react";
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
  const [justAdded, setJustAdded] = useState(false);

  return (
    <Button
      type="button"
      variant={inWatchlist ? "secondary" : "outline"}
      size="sm"
      data-umami-event={inWatchlist ? "watchlist-remove" : "watchlist-add"}
      className="transition-colors"
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
          setJustAdded(true);
        }
      }}
    >
      <Star
        className={`transition-transform ${inWatchlist ? "fill-current" : ""} ${justAdded ? "animate-pop" : ""}`}
        data-icon="inline-start"
        onAnimationEnd={() => setJustAdded(false)}
      />
      {inWatchlist ? "In Watchlist" : "Add to Watchlist"}
    </Button>
  );
}
