import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  poweredByHeader: false,
  experimental: {
    // The one stylesheet <link> Next emits is render-blocking, and on a
    // single VPS behind Caddy that extra round trip cost ~230ms of LCP for
    // first-time visitors — which is most of the traffic here, since it
    // arrives from search. Tailwind's atomic output is small enough (~10KB)
    // that shipping it inside the HTML beats a separately cached file.
    // Production-only; dev still uses <link> tags.
    inlineCss: true,
  },
  turbopack: {
    root: path.resolve(__dirname),
  },
  allowedDevOrigins: ["localhost", "127.0.0.1"],
  images: {
    remotePatterns: [
      { protocol: "https", hostname: "logos.stocktwits-cdn.com" },
      { protocol: "https", hostname: "images.unsplash.com" },
    ],
  },
  async redirects() {
    return [
      // companySlug("S&P 500") moved from "sentiment-of-s-p-500-stock" to
      // "sentiment-of-sp-500-stock" (see slugify() in lib/utils/tickers.ts) — this page
      // already ranks for real search queries, so redirect rather than let it 404.
      {
        source: "/blog/sentiment-of-s-p-500-stock",
        destination: "/blog/sentiment-of-sp-500-stock",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
