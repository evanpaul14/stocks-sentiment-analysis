"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { GoogleIcon } from "@/components/icons/GoogleIcon";
import { mergeLocalDataIntoAccount } from "@/lib/auth/mergeLocalData";
import { notifySessionChanged } from "@/lib/auth/sessionEvents";
import { useSession } from "@/lib/auth/useSession";

export default function LoginPage() {
  const router = useRouter();
  const { loggedIn } = useSession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "error" | "unverified">("idle");

  useEffect(() => {
    if (loggedIn) router.replace("/account");
  }, [loggedIn, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (response.ok) {
        notifySessionChanged();
        await mergeLocalDataIntoAccount();
        router.push("/account");
        router.refresh();
        return;
      }
      const data = await response.json().catch(() => null);
      setStatus(data?.error === "email_not_verified" ? "unverified" : "error");
    } catch {
      setStatus("error");
    }
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Sign in</h1>

      <a
        href="/api/auth/google/start"
        className="mb-4 flex w-full items-center justify-center gap-3 rounded-lg border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted/50"
      >
        <GoogleIcon className="size-4.5" />
        Continue with Google
      </a>

      <div className="mb-4 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="h-px flex-1 bg-border" />
        or
        <span className="h-px flex-1 bg-border" />
      </div>

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
        <input
          type="password"
          required
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
        />
        <Button type="submit" disabled={status === "loading"} className="w-full">
          {status === "loading" ? "Signing in…" : "Sign in"}
        </Button>

        {status === "error" && (
          <p className="text-xs text-destructive">Invalid email or password.</p>
        )}
        {status === "unverified" && (
          <p className="text-xs text-destructive">
            Verify your email before signing in — check your inbox for the link.
          </p>
        )}
      </form>

      <p className="mt-4 text-sm text-muted-foreground">
        <Link href="/reset-password" className="underline">
          Forgot your password?
        </Link>
      </p>
      <p className="mt-2 text-sm text-muted-foreground">
        No account?{" "}
        <Link href="/signup" className="underline">
          Sign up
        </Link>
      </p>
    </main>
  );
}
