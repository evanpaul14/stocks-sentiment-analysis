"use client";

import Link from "next/link";
import { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { PasswordRequirementsList } from "@/components/auth/PasswordRequirementsList";
import { isPasswordValid } from "@/lib/auth/passwordPolicy";
import { useSession } from "@/lib/auth/useSession";

export default function ResetPasswordPage() {
  return (
    <Suspense
      fallback={
        <main className="mx-auto max-w-sm px-4 py-10">
          <p className="text-sm text-muted-foreground">Loading…</p>
        </main>
      }
    >
      <ResetPasswordForm />
    </Suspense>
  );
}

function ResetPasswordForm() {
  const router = useRouter();
  const { loggedIn } = useSession();
  const token = useSearchParams().get("token");

  useEffect(() => {
    if (loggedIn) router.replace("/account");
  }, [loggedIn, router]);

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Reset password</h1>
      {token ? <NewPasswordForm token={token} /> : <RequestResetForm />}
    </main>
  );
}

function RequestResetForm() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done">("idle");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    try {
      await fetch("/api/auth/request-password-reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
    } finally {
      setStatus("done");
    }
  }

  if (status === "done") {
    return (
      <p className="text-sm text-muted-foreground">
        If an account exists for <strong>{email}</strong>, we sent a password reset link.
      </p>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <input
        type="email"
        required
        autoComplete="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="you@example.com"
        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
      />
      <Button type="submit" disabled={status === "loading"} className="w-full">
        {status === "loading" ? "Sending…" : "Send reset link"}
      </Button>
      <p className="text-sm text-muted-foreground">
        <Link href="/login" className="underline">
          Back to sign in
        </Link>
      </p>
    </form>
  );
}

function NewPasswordForm({ token }: { token: string }) {
  const [newPassword, setNewPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    try {
      const response = await fetch("/api/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, newPassword }),
      });
      setStatus(response.ok ? "done" : "error");
    } catch {
      setStatus("error");
    }
  }

  if (status === "done") {
    return (
      <p className="text-sm text-muted-foreground">
        Password updated.{" "}
        <Link href="/login" className="underline">
          Sign in
        </Link>
        .
      </p>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="space-y-2">
        <input
          type="password"
          required
          autoComplete="new-password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          placeholder="New password"
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
        />
        <PasswordRequirementsList password={newPassword} />
      </div>
      <Button
        type="submit"
        disabled={status === "loading" || !isPasswordValid(newPassword)}
        className="w-full"
      >
        {status === "loading" ? "Updating…" : "Update password"}
      </Button>
      {status === "error" && (
        <p className="text-xs text-destructive">Invalid or expired link — request a new one.</p>
      )}
    </form>
  );
}
