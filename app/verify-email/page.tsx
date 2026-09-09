"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { mergeLocalDataIntoAccount } from "@/lib/auth/mergeLocalData";
import { notifySessionChanged } from "@/lib/auth/sessionEvents";

export default function VerifyEmailPage() {
  return (
    <Suspense
      fallback={
        <main className="mx-auto max-w-sm px-4 py-10">
          <p className="text-sm text-muted-foreground">Verifying…</p>
        </main>
      }
    >
      <VerifyEmailForm />
    </Suspense>
  );
}

function VerifyEmailForm() {
  const searchParams = useSearchParams();
  const token = searchParams.get("token");
  const [status, setStatus] = useState<"loading" | "done" | "error">(
    token ? "loading" : "error"
  );

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    fetch("/api/auth/verify-email", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    })
      .then(async (response) => {
        if (cancelled) return;
        if (response.ok) {
          notifySessionChanged();
          await mergeLocalDataIntoAccount();
          setStatus("done");
        } else {
          setStatus("error");
        }
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      {status === "loading" && <p className="text-sm text-muted-foreground">Verifying…</p>}
      {status === "done" && (
        <>
          <h1 className="mb-2 text-2xl font-semibold">Email verified</h1>
          <p className="text-sm text-muted-foreground">
            You&apos;re signed in.{" "}
            <Link href="/account" className="underline">
              Go to your account
            </Link>
            .
          </p>
        </>
      )}
      {status === "error" && (
        <>
          <h1 className="mb-2 text-2xl font-semibold">Invalid or expired link</h1>
          <p className="text-sm text-muted-foreground">
            Request a new verification email from the{" "}
            <Link href="/login" className="underline">
              sign in
            </Link>{" "}
            page.
          </p>
        </>
      )}
    </main>
  );
}
