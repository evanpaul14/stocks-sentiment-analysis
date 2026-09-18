import { NextResponse, type NextRequest } from "next/server";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

/**
 * Only a same-origin relative path is a valid `next` — Supabase's redirect
 * allow-list only covers the base /auth/callback URL, not this query param,
 * so an absolute or protocol-relative value here (e.g. `next=https://evil.com`
 * or `next=//evil.com`) would otherwise send a just-authenticated user
 * straight to an attacker's site.
 */
function sanitizeNext(next: string | null): string {
  if (!next || !next.startsWith("/") || next.startsWith("//") || next.startsWith("/\\")) {
    return "/account";
  }
  return next;
}

/**
 * Single generic landing point for both Google OAuth and password-recovery
 * links — Supabase's own redirect always carries a `?code=` (PKCE), whether
 * it's an OAuth sign-in or a recovery email link. `next` lets callers pick
 * where to land afterward (e.g. /reset-password for recovery).
 */
export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code");
  const next = sanitizeNext(request.nextUrl.searchParams.get("next"));

  // Built from SITE_BASE_URL, not request.url — behind the Caddy reverse
  // proxy, request.url reflects the internal localhost origin Node is
  // bound to, not the public domain.
  const destination = new URL(next, process.env.SITE_BASE_URL ?? "http://localhost:3000");

  if (!code) {
    const failure = new URL("/login", process.env.SITE_BASE_URL ?? "http://localhost:3000");
    failure.searchParams.set("error", "auth_callback_failed");
    return NextResponse.redirect(failure);
  }

  const response = NextResponse.redirect(destination);
  const supabase = createClient(request, response);
  const { error } = await supabase.auth.exchangeCodeForSession(code);

  if (error) {
    console.error("[auth/callback] exchangeCodeForSession failed", error);
    const failure = new URL("/login", process.env.SITE_BASE_URL ?? "http://localhost:3000");
    failure.searchParams.set("error", "auth_callback_failed");
    return NextResponse.redirect(failure);
  }

  return response;
}
