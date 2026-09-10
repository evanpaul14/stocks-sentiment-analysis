import Link from "next/link";

const FOOTER_LINKS = [
  { href: "/about", label: "About" },
  { href: "/contact", label: "Contact" },
  { href: "/privacy", label: "Privacy" },
  { href: "/blog", label: "Blog" },
];

export function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto max-w-4xl px-4 py-8 sm:max-w-none sm:px-6 lg:px-10">
        <p className="max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Stock Sentiment provides automated, AI-generated market data and sentiment analysis for
          informational purposes only — it is not financial, investment, or trading advice.
        </p>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <nav className="flex flex-wrap gap-x-4 gap-y-2 text-sm">
            {FOOTER_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                {link.label}
              </Link>
            ))}
          </nav>
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} Stock Sentiment
          </p>
        </div>
      </div>
    </footer>
  );
}
