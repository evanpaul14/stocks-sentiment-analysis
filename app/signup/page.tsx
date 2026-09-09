"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { GoogleIcon } from "@/components/icons/GoogleIcon";
import { PasswordRequirementsList } from "@/components/auth/PasswordRequirementsList";
import { isPasswordValid } from "@/lib/auth/passwordPolicy";
import { useSession } from "@/lib/auth/useSession";

export default function SignupPage() {
  const router = useRouter();
  const { loggedIn } = useSession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error" | "taken">("idle");

  useEffect(() => {
    if (loggedIn) router.replace("/account");
  }, [loggedIn, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    try {
      const response = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      if (response.ok) {
        setStatus("done");
        return;
      }
      setStatus(response.status === 409 ? "taken" : "error");
    } catch {
      setStatus("error");
    }
  }

  if (status === "done") {
    return (
      <main className="mx-auto max-w-sm px-4 py-10">
        <h1 className="mb-2 text-2xl font-semibold">Check your email</h1>
        <p className="text-sm text-muted-foreground">
          We sent a verification link to <strong>{email}</strong>. Click it to finish
          setting up your account.
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Sign up</h1>

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
        <div className="space-y-2">
          <input
            type="password"
            required
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
          />
          <PasswordRequirementsList password={password} />
        </div>
        <Button
          type="submit"
          disabled={status === "loading" || !isPasswordValid(password)}
          className="w-full"
        >
          {status === "loading" ? "Creating account…" : "Sign up"}
        </Button>

        {status === "error" && (
          <p className="text-xs text-destructive">Something went wrong — try again.</p>
        )}
        {status === "taken" && (
          <p className="text-xs text-destructive">
            An account with that email already exists.{" "}
            <Link href="/login" className="underline">
              Sign in
            </Link>{" "}
            instead.
          </p>
        )}
      </form>

      <p className="mt-4 text-sm text-muted-foreground">
        Already have an account?{" "}
        <Link href="/login" className="underline">
          Sign in
        </Link>
      </p>
    </main>
  );
}
