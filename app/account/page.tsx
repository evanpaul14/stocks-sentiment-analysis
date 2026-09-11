"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { mergeLocalDataIntoAccount } from "@/lib/auth/mergeLocalData";
import { createClient } from "@/lib/supabase/client";

interface AccountInfo {
  email: string | null;
  emailConfirmedAt: string | null;
  providers: string[];
}

export default function AccountPage() {
  const router = useRouter();
  const [user, setUser] = useState<AccountInfo | null>(null);
  const [status, setStatus] = useState<"loading" | "loaded" | "unauthenticated">("loading");
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [subscription, setSubscription] = useState<
    "loading" | "subscribed" | "unsubscribed" | "unavailable"
  >("loading");
  const [subscriptionSaving, setSubscriptionSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (cancelled) return;
      if (!data.user) {
        setStatus("unauthenticated");
        return;
      }
      setUser({
        email: data.user.email ?? null,
        emailConfirmedAt: data.user.email_confirmed_at ?? null,
        providers: (data.user.identities ?? []).map((i) => i.provider),
      });
      setStatus("loaded");
      // Idempotent — covers landing here fresh from the Google OAuth
      // redirect, which can't run client-side merge logic itself.
      mergeLocalDataIntoAccount();
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (status !== "loaded") return;
    let cancelled = false;
    fetch("/api/account/market-summary-subscription")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled) return;
        setSubscription(
          data && typeof data.subscribed === "boolean"
            ? data.subscribed
              ? "subscribed"
              : "unsubscribed"
            : "unavailable"
        );
      })
      .catch(() => {
        if (!cancelled) setSubscription("unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [status]);

  async function handleToggleSubscription() {
    if (subscription !== "subscribed" && subscription !== "unsubscribed") return;
    const nextSubscribed = subscription === "unsubscribed";
    setSubscriptionSaving(true);
    try {
      const response = await fetch("/api/account/market-summary-subscription", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subscribed: nextSubscribed }),
      });
      if (response.ok) {
        setSubscription(nextSubscribed ? "subscribed" : "unsubscribed");
      }
    } finally {
      setSubscriptionSaving(false);
    }
  }

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  async function handleSignOut() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  async function handleDelete() {
    await fetch("/api/account", { method: "DELETE" });
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/");
    router.refresh();
  }

  if (status !== "loaded" || !user) {
    return (
      <main className="mx-auto max-w-sm px-4 py-10">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </main>
    );
  }

  const providers = new Set(user.providers);

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Account</h1>

      <dl className="mb-8 space-y-3 text-sm">
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Email</dt>
          <dd className="truncate font-medium">{user.email}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Status</dt>
          <dd>{user.emailConfirmedAt ? "Verified" : "Not verified"}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Sign-in methods</dt>
          <dd>
            {[providers.has("email") && "Password", providers.has("google") && "Google"]
              .filter(Boolean)
              .join(", ")}
          </dd>
        </div>
      </dl>

      {subscription !== "unavailable" && (
        <div className="mb-8 flex items-center justify-between gap-4 rounded-lg border p-3">
          <div>
            <p className="text-sm font-medium">Market summary emails</p>
            <p className="text-xs text-muted-foreground">
              Daily wrap-up of what moved the market, sent to {user.email}.
            </p>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={subscription === "loading" || subscriptionSaving}
            onClick={handleToggleSubscription}
          >
            {subscription === "loading"
              ? "Loading…"
              : subscription === "subscribed"
                ? "Unsubscribe"
                : "Subscribe"}
          </Button>
        </div>
      )}

      <div className="space-y-3">
        <Button type="button" variant="outline" className="w-full" onClick={handleSignOut}>
          Sign out
        </Button>

        {!confirmingDelete ? (
          <Button
            type="button"
            variant="ghost"
            className="w-full text-destructive hover:text-destructive"
            onClick={() => setConfirmingDelete(true)}
          >
            Delete account
          </Button>
        ) : (
          <div className="rounded-lg border border-destructive/30 p-3">
            <p className="mb-3 text-xs text-muted-foreground">
              This permanently deletes your account, watchlist, and search history. This
              can&apos;t be undone.
            </p>
            <div className="flex gap-2">
              <Button type="button" variant="destructive" onClick={handleDelete} className="flex-1">
                Confirm delete
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => setConfirmingDelete(false)}
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
