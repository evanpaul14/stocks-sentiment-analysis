import { NextResponse, type NextRequest } from "next/server";
import { createClient } from "@/lib/supabase/server";
import { getClientIp } from "@/lib/ratelimit/withRateLimit";
import { getLimiter } from "@/lib/ratelimit/tokenBucket";
import { isVerifiedCrawler } from "@/lib/ratelimit/verifiedCrawler";

// Site-wide per-IP page limit: 3 requests/second sustained. The bucket holds
// 20 tokens (20 per 6.67s window = 3/s refill) so a normal page load and quick
// click-through never trips it, but a scraper hammering dozens of pages a
// second is cut off within a few seconds. API routes keep their own limits.
const PAGE_LIMIT = 20;
const PAGE_WINDOW_MS = Math.round((PAGE_LIMIT / 3) * 1000);
const pageLimiter = getLimiter("pages", PAGE_LIMIT, PAGE_WINDOW_MS);

export async function proxy(request: NextRequest) {
  // Link prefetches fire in bursts (one per visible link) with no user
  // intent, and /api routes are already limited per route, so neither
  // counts against the page limit.
  const isPrefetch = request.headers.has("next-router-prefetch");
  const isApi = request.nextUrl.pathname.startsWith("/api/");
  if (!isPrefetch && !isApi) {
    const ip = getClientIp(request);
    if (!pageLimiter.consume(`pages:${ip}`)) {
      // Verified search crawlers (Googlebot etc.) share IPs and must never be
      // throttled. Checked only after the bucket is empty so normal traffic
      // never pays for the reverse-DNS lookup.
      const crawler = await isVerifiedCrawler(ip, request.headers.get("user-agent"));
      if (!crawler) {
        return new NextResponse("Too many requests", {
          status: 429,
          headers: { "retry-after": "5", "content-type": "text/plain" },
        });
      }
    }
  }

  const nonce = Buffer.from(crypto.randomUUID()).toString("base64");

  // Derived from UMAMI_SCRIPT_URL so the CSP can never drift out of sync with
  // wherever analytics actually points (cloud vs. self-hosted). Empty string
  // when analytics is unconfigured, which collapses to no extra source.
  //
  // Umami Cloud spreads itself across hosts — the script comes from
  // cloud.umami.is but events post to gateway.umami.is — so allow the whole
  // umami.is domain rather than chasing individual subdomains as they change.
  // (`*.umami.is` doesn't match the bare apex, hence both.) Self-hosted
  // installs just get their own single origin.
  const analyticsOrigins = (() => {
    const url = process.env.UMAMI_SCRIPT_URL;
    if (!url) return "";
    try {
      const { origin, hostname } = new URL(url);
      if (hostname === "umami.is" || hostname.endsWith(".umami.is")) {
        return "https://umami.is https://*.umami.is";
      }
      // A first-party proxied path is already covered by 'self'; naming the
      // origin again in that case is redundant but harmless, so don't
      // special-case it.
      return origin;
    } catch {
      return "";
    }
  })();

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
    // Umami is the same story: the script tag loads fine under script-src,
    // but its pageview beacon to /api/send is a connect-src request, so
    // leaving the analytics origin out here silently drops every pageview.
    `connect-src 'self' ${process.env.NEXT_PUBLIC_SUPABASE_URL ?? ""} https://challenges.cloudflare.com ${analyticsOrigins}`,
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
