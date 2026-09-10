import type { Metadata } from "next";
import { ContactForm } from "@/components/layout/ContactForm";

export const metadata: Metadata = {
  title: "Contact",
  description: "Get in touch with the Stock Sentiment team.",
  alternates: { canonical: "/contact" },
};

export default function ContactPage() {
  return (
    <main className="mx-auto max-w-md px-4 py-10">
      <h1 className="mb-2 text-2xl font-semibold">Contact</h1>
      <p className="mb-6 text-sm text-muted-foreground">
        Questions, corrections, or feedback about the site or its data — send a message and
        we&apos;ll get back to you.
      </p>
      <ContactForm />
    </main>
  );
}
