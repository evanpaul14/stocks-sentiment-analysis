"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

const INTRO_SEEN_KEY = "search-limit-intro-seen";

interface SearchLimitPromptProps {
  used: number;
  limit: number;
  limited: boolean;
  /** Page content shown behind the prompt; hidden while the limit popup is open. */
  children?: React.ReactNode;
}

/**
 * Signup nudge for anonymous visitors: shown once after their first search,
 * then again whenever they've hit the free-search limit.
 */
export function SearchLimitPrompt({ used, limit, limited, children }: SearchLimitPromptProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [closed, setClosed] = useState(false);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    let seen = false;
    try {
      seen = localStorage.getItem(INTRO_SEEN_KEY) === "1";
    } catch {
      // Storage blocked — treat as unseen; worst case the intro repeats.
    }
    if (dialog.open || !(limited || !seen)) return;
    // At the limit the dialog is non-modal so the header stays usable; the
    // locked page content is simply not rendered while it's open.
    if (limited) dialog.show();
    else dialog.showModal();
  }, [limited]);

  function dismiss() {
    if (!limited) {
      try {
        localStorage.setItem(INTRO_SEEN_KEY, "1");
      } catch {
        // ignore
      }
    }
    dialogRef.current?.close();
    setClosed(true);
  }

  const remaining = Math.max(limit - used, 0);

  return (
    <>
      {(!limited || closed) && children}
      {limited && closed && <LockedNotice />}
      <dialog
        ref={dialogRef}
        onCancel={dismiss}
        onClick={(e) => {
          if (e.target === dialogRef.current) dismiss();
        }}
        aria-labelledby="search-limit-title"
        className={`m-auto w-[min(92vw,26rem)] rounded-none border border-border bg-card p-6 text-card-foreground shadow-lg ${
          // Non-modal at the limit: pinned below the 4rem header, centered in the rest of the viewport.
          limited ? "fixed inset-x-0 top-16 bottom-0 h-fit" : "backdrop:bg-black/50"
        }`}
      >
        <h2 id="search-limit-title" className="text-lg font-semibold">
          {limited ? "You've used all your free searches" : "Create a free account"}
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          {limited
            ? `Visitors without an account get ${limit} stock searches per day. Sign up for free to search without limits.`
            : `You have ${remaining} of ${limit} free searches left today. Sign up for free to search without limits, plus a saved watchlist and search history across devices.`}
        </p>
        <div className="mt-5 flex flex-wrap items-center gap-2">
          <Link
            href="/signup"
            data-umami-event={limited ? "search-limit-signup" : "search-intro-signup"}
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
          <Button type="button" variant="ghost" onClick={dismiss} className="ml-auto">
            {limited ? "Close" : "Maybe later"}
          </Button>
        </div>
      </dialog>
    </>
  );
}

function LockedNotice() {
  return (
    <section className="rounded-none border border-border bg-card p-6">
      <h2 className="text-lg font-medium">Free search limit reached</h2>
      <p className="mt-2 text-sm text-muted-foreground">
        Sign up for free to search without limits, or come back tomorrow.
      </p>
      <div className="mt-4 flex gap-2">
        <Link
          href="/signup"
          className="inline-flex h-8 pointer-coarse:h-11 items-center bg-primary px-3 text-sm font-medium text-primary-foreground hover:bg-primary/80"
        >
          Sign up free
        </Link>
        <Link
          href="/login"
          className="inline-flex h-8 pointer-coarse:h-11 items-center border border-border bg-background px-3 text-sm font-medium hover:bg-muted"
        >
          Log in
        </Link>
      </div>
    </section>
  );
}
