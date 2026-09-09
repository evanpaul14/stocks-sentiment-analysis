import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Reset password",
  robots: { index: false, follow: true },
};

export default function ResetPasswordLayout({ children }: LayoutProps<"/reset-password">) {
  return children;
}
