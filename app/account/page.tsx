"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import type { User } from "@supabase/supabase-js";
import { Button } from "@/components/ui/button";
import { mergeLocalDataIntoAccount } from "@/lib/auth/mergeLocalData";
import { createClient } from "@/lib/supabase/client";

export default function AccountPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [status, setStatus] = useState<"loading" | "loaded" | "unauthenticated">("loading");
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      if (cancelled) return;
      if (!data.user) {
        setStatus("unauthenticated");
        return;
      }
      setUser(data.user);
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

  const providers = new Set((user.identities ?? []).map((i) => i.provider));

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
          <dd>{user.email_confirmed_at ? "Verified" : "Not verified"}</dd>
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
