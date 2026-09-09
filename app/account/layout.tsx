import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Account",
  robots: { index: false, follow: true },
};

export default function AccountLayout({ children }: LayoutProps<"/account">) {
  return children;
}
