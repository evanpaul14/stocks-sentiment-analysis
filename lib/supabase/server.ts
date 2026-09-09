import { createServerClient, parseCookieHeader } from "@supabase/ssr";
import type { NextRequest, NextResponse } from "next/server";

/**
 * Full read+write client — only needed where a route establishes or
 * changes the session itself (e.g. the OAuth/recovery callback). Writes
 * any refreshed/new cookies onto `response`.
 */
export function createClient(request: NextRequest, response: NextResponse) {
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return parseCookieHeader(request.headers.get("cookie") ?? "");
        },
        setAll(cookiesToSet) {
          cookiesToSet.forEach(({ name, value, options }) => {
            response.cookies.set(name, value, options);
          });
        },
      },
    }
  );
}

/** Read-only client — for routes that only need to know who's calling; session refresh happens in proxy.ts. */
export function createReadOnlyClient(request: NextRequest) {
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return parseCookieHeader(request.headers.get("cookie") ?? "");
        },
        setAll() {
          // No-op: this client is only used to read the current user.
        },
      },
    }
  );
}
