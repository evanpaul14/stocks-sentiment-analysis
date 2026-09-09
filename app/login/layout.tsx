import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Sign in",
  robots: { index: false, follow: true },
};

export default function LoginLayout({ children }: LayoutProps<"/login">) {
  return children;
}
