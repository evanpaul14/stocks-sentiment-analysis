import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Watchlist",
  robots: { index: false, follow: true },
};

export default function WatchlistLayout({ children }: LayoutProps<"/watchlist">) {
  return children;
}
