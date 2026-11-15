"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { INTRO_SEEN_KEY } from "@/components/stock/SearchLimitPrompt";
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
  const { add, remove, has, loggedIn, isLoading } = useWatchlist();
  const inWatchlist = has(symbol);
  const [justAdded, setJustAdded] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);

  function addEntry() {
    add({ symbol, companyName, lastPrice: price, lastChangePercent: changePercent });
    setJustAdded(true);
  }

  function handleClick() {
    if (inWatchlist) {
      remove(symbol);
      return;
    }
    if (isLoading) return; // session not resolved yet; don't guess logged-out
    if (loggedIn) {
      addEntry();
      return;
    }
    // Anonymous: explain the account benefit before falling back to device-only storage.
    try {
      // The user has now seen a signup pitch; don't stack the intro popup on top.
      localStorage.setItem(INTRO_SEEN_KEY, "1");
    } catch {
      // ignore
    }
    dialogRef.current?.showModal();
  }

  return (
    <>
      <Button
        type="button"
        variant={inWatchlist ? "secondary" : "outline"}
        size="sm"
        data-umami-event={inWatchlist ? "watchlist-remove" : "watchlist-add"}
        className="transition-colors"
        onClick={handleClick}
      >
        <Star
          className={`transition-transform ${inWatchlist ? "fill-current" : ""} ${justAdded ? "animate-pop" : ""}`}
          data-icon="inline-start"
          onAnimationEnd={() => setJustAdded(false)}
        />
        {inWatchlist ? "In Watchlist" : "Add to Watchlist"}
      </Button>
      <dialog
        ref={dialogRef}
        onClick={(e) => {
          if (e.target === dialogRef.current) dialogRef.current?.close();
        }}
        aria-labelledby="watchlist-account-title"
        className="m-auto w-[min(92vw,26rem)] rounded-none border border-border bg-card p-6 text-card-foreground shadow-lg backdrop:bg-black/50"
      >
        <h2 id="watchlist-account-title" className="text-lg font-semibold">
          Your watchlist works best with an account
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Sign up for free to keep {symbol} and the rest of your watchlist synced across all your
          devices. Without an account it&apos;s only saved in this browser and can be lost if you
          clear your data.
        </p>
        <div className="mt-5 flex flex-wrap items-center gap-2">
          <Link
            href="/signup"
            data-umami-event="watchlist-prompt-signup"
            className="inline-flex h-8 pointer-coarse:h-11 items-center justify-center bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/80"
          >
            Sign up free
          </Link>
          <Link
            href="/login"
            className="inline-flex h-8 pointer-coarse:h-11 items-center justify-center border border-border bg-background px-3 text-sm font-medium hover:bg-muted"
          >
            Log in
          </Link>
          <Button
            type="button"
            variant="ghost"
            data-umami-event="watchlist-prompt-local"
            className="ml-auto"
            onClick={() => {
              addEntry();
              dialogRef.current?.close();
            }}
          >
            Save on this device only
          </Button>
        </div>
      </dialog>
    </>
  );
}
