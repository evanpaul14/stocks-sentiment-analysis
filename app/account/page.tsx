"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { mergeLocalDataIntoAccount } from "@/lib/auth/mergeLocalData";
import { notifySessionChanged } from "@/lib/auth/sessionEvents";

interface AccountInfo {
  email: string;
  emailVerified: boolean;
  hasPassword: boolean;
  hasGoogle: boolean;
}

export default function AccountPage() {
  const router = useRouter();
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [status, setStatus] = useState<"loading" | "loaded" | "unauthenticated">("loading");
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/account")
      .then(async (res) => {
        if (cancelled) return;
        if (!res.ok) {
          setStatus("unauthenticated");
          return;
        }
        setAccount(await res.json());
        setStatus("loaded");
        // Idempotent — covers landing here fresh from the Google OAuth
        // redirect, which can't run client-side merge logic itself.
        mergeLocalDataIntoAccount();
      })
      .catch(() => {
        if (!cancelled) setStatus("unauthenticated");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  async function handleSignOut() {
    await fetch("/api/auth/logout", { method: "POST" });
    notifySessionChanged();
    router.push("/");
    router.refresh();
  }

  async function handleDelete() {
    await fetch("/api/account", { method: "DELETE" });
    notifySessionChanged();
    router.push("/");
    router.refresh();
  }

  if (status !== "loaded" || !account) {
    return (
      <main className="mx-auto max-w-sm px-4 py-10">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Account</h1>

      <dl className="mb-8 space-y-3 text-sm">
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Email</dt>
          <dd className="truncate font-medium">{account.email}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Status</dt>
          <dd>{account.emailVerified ? "Verified" : "Not verified"}</dd>
        </div>
        <div className="flex justify-between gap-4">
          <dt className="text-muted-foreground">Sign-in methods</dt>
          <dd>
            {[account.hasPassword && "Password", account.hasGoogle && "Google"]
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
