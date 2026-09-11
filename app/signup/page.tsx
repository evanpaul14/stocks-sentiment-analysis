"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { GoogleIcon } from "@/components/icons/GoogleIcon";
import { PasswordRequirementsList } from "@/components/auth/PasswordRequirementsList";
import { isPasswordValid } from "@/lib/auth/passwordPolicy";
import { useSession } from "@/lib/auth/useSession";
import { createClient } from "@/lib/supabase/client";

export default function SignupPage() {
  const router = useRouter();
  const { loggedIn } = useSession();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle");

  useEffect(() => {
    if (loggedIn) router.replace("/account");
  }, [loggedIn, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    const supabase = createClient();
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
    });
    // Supabase intentionally doesn't distinguish "already registered" from
    // success here (anti-enumeration) when email confirmations are on, so
    // this always shows the same "check your email" state on success.
    setStatus(error ? "error" : "done");
  }

  async function handleGoogle() {
    const supabase = createClient();
    await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
  }

  if (status === "done") {
    return (
      <main className="mx-auto max-w-sm px-4 py-10">
        <h1 className="mb-2 text-2xl font-semibold">Check your email</h1>
        <p role="status" className="text-sm text-muted-foreground">
          If <strong>{email}</strong> isn&apos;t already registered, we sent a confirmation
          link. Click it to finish setting up your account.
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-sm px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Sign up</h1>

      <button
        type="button"
        onClick={handleGoogle}
        className="mb-4 flex w-full items-center justify-center gap-3 rounded-lg border border-border px-3 py-2 text-sm font-medium transition-colors duration-150 hover:bg-muted/50"
      >
        <GoogleIcon className="size-4.5" />
        Continue with Google
      </button>

      <div className="mb-4 flex items-center gap-3 text-xs text-muted-foreground">
        <span className="h-px flex-1 bg-border" />
        or
        <span className="h-px flex-1 bg-border" />
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <label htmlFor="signup-email" className="sr-only">
          Email
        </label>
        <input
          id="signup-email"
          type="email"
          required
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="you@example.com"
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-ring focus:ring-3 focus:ring-ring/50"
        />
        <div className="space-y-2">
          <label htmlFor="signup-password" className="sr-only">
            Password
          </label>
          <input
            id="signup-password"
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

        <div role="alert" aria-live="assertive">
          {status === "error" && (
            <p className="text-xs text-destructive">Something went wrong — try again.</p>
          )}
        </div>
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
