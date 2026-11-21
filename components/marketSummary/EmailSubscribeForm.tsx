"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function EmailSubscribeForm() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    try {
      const response = await fetch("/api/market-summary/subscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      setStatus(response.ok ? "done" : "error");
    } catch {
      setStatus("error");
    }
  }

  if (status === "done") {
    return (
      // The form (and its focused submit button) unmounts on success: move
      // focus here so keyboard and screen-reader users land on the result.
      <p
        role="status"
        tabIndex={-1}
        ref={(el) => el?.focus()}
        className="text-sm text-[var(--color-chart-1)] outline-none"
      >
        You&apos;re subscribed — check your inbox.
      </p>
    );
  }

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="market-summary-email" className="mb-1 block text-sm font-medium">
        Email
      </label>
      <div className="flex gap-2">
        <input
          id="market-summary-email"
          type="email"
          required
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          className="min-w-0 flex-1 rounded-none border border-field bg-background px-3 py-2 text-sm pointer-coarse:min-h-11 pointer-coarse:text-base outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
        />
        <Button type="submit" disabled={status === "loading"}>
          {status === "loading" ? "Subscribing…" : "Subscribe"}
        </Button>
      </div>
      <div role="alert" aria-live="assertive">
        {status === "error" && (
          <p className="mt-2 text-xs text-destructive-text">Something went wrong — try again.</p>
        )}
      </div>
    </form>
  );
}
