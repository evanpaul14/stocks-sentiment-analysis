import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Stock Sentiment",
    short_name: "Stock Sentiment",
    description:
      "Real-time stock prices, historical charts, and AI-powered news sentiment analysis.",
    start_url: "/",
    display: "standalone",
    background_color: "#211d19",
    theme_color: "#211d19",
    icons: [
      { src: "/logo-icon-512.png", sizes: "512x512", type: "image/png" },
    ],
  };
}
