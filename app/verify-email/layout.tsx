import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Verify email",
  robots: { index: false, follow: true },
};

export default function VerifyEmailLayout({ children }: LayoutProps<"/verify-email">) {
  return children;
}
