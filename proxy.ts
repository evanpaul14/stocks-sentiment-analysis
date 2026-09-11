import { NextResponse, type NextRequest } from "next/server";
import { createClient } from "@/lib/supabase/server";

export async function proxy(request: NextRequest) {
  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");

  const csp = [
    `default-src 'self'`,
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'`,
    `style-src 'self' 'unsafe-inline'`,
    `img-src 'self' data: https:`,
    `font-src 'self' data:`,
    // Turnstile's challenge renders in an iframe from Cloudflare's domain —
    // without this it silently falls back to default-src 'self' and the
    // widget never appears (the script loads fine, only the iframe is blocked).
    `frame-src https://challenges.cloudflare.com`,
    // Supabase's client SDK talks to the project's own domain (auth, token
    // refresh) — CSP's connect-src is same-origin-only by default, so it
    // has to be explicitly allowed here or every Supabase call is blocked.
    `connect-src 'self' ${process.env.NEXT_PUBLIC_SUPABASE_URL ?? ""} https://challenges.cloudflare.com`,
    `frame-ancestors 'none'`,
    `base-uri 'self'`,
    `form-action 'self'`,
  ].join("; ");

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("content-security-policy", csp);

  const response = NextResponse.next({
    request: { headers: requestHeaders },
  });

  // Refreshes the Supabase session cookie (if needed) before any route
  // handler or page runs, per @supabase/ssr's documented proxy/middleware
  // pattern — keeps signed-in users signed in across access-token expiry.
  const supabase = createClient(request, response);
  await supabase.auth.getClaims();

  response.headers.set("content-security-policy", csp);
  response.headers.set("x-content-type-options", "nosniff");
  response.headers.set(
    "referrer-policy",
    process.env.SECURITY_HEADERS_REFERRER_POLICY ??
      "strict-origin-when-cross-origin"
  );
  response.headers.set(
    "permissions-policy",
    process.env.SECURITY_HEADERS_PERMISSIONS_POLICY ??
      "camera=(), microphone=(), geolocation=(), interest-cohort=()"
  );

  return response;
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico)$).*)",
  ],
};
