import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  poweredByHeader: false,
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
