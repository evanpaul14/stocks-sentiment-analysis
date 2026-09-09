"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, Search, User, X } from "lucide-react";
import { SearchBar } from "@/components/search/SearchBar";
import { buttonVariants } from "@/components/ui/button";
import { useSession } from "@/lib/auth/useSession";

const NAV_LINKS = [
  { href: "/watchlist", label: "Watchlist" },
  { href: "/trending", label: "Trending" },
  { href: "/market-summary", label: "Market Summary" },
  { href: "/blog", label: "Blog" },
];

export function Header() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const pathname = usePathname();
  const { loggedIn } = useSession();

  // Adjust state during render rather than resetting it in an effect —
  // avoids an extra render pass on navigation (see React docs: "Adjusting
  // state when a prop changes").
  const [lastPathname, setLastPathname] = useState(pathname);
  if (pathname !== lastPathname) {
    setLastPathname(pathname);
    setIsMenuOpen(false);
    setIsSearchOpen(false);
  }

  useEffect(() => {
    // Hysteresis: shrinking the header changes its height, which shifts
    // scrollY across a single fixed threshold and toggles it right back
    // (visible as bouncing). Separate enter/exit points give it a dead
    // zone wider than that shift so it can't retrigger itself.
    function handleScroll() {
      setIsScrolled((wasScrolled) =>
        wasScrolled ? window.scrollY > 4 : window.scrollY > 24,
      );
    }
    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur transition-[padding] duration-200 ${
        isScrolled ? "py-2" : "py-4"
      }`}
    >
      <div className="mx-auto flex max-w-4xl items-center justify-between gap-4 px-4 sm:max-w-none sm:px-6 lg:px-10">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 whitespace-nowrap font-serif text-lg tracking-tight text-foreground"
        >
          <Image
            src="/logo-mark.png"
            alt=""
            width={28}
            height={28}
            priority
            className="size-7"
          />
          Stock Sentiment
        </Link>

        <div className="hidden min-w-0 flex-1 lg:block">
          <SearchBar showSuggestions className="" />
        </div>

        <div className="flex shrink-0 items-center gap-4 lg:gap-6">
          <nav className="hidden items-center gap-4 whitespace-nowrap text-sm sm:flex lg:gap-6">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`transition-colors ${
                  pathname === link.href
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {link.label}
              </Link>
            ))}
            {loggedIn ? (
              <Link
                href="/account"
                aria-label="Account"
                className="text-muted-foreground transition-colors hover:text-foreground [&_svg]:transition-transform [&_svg]:duration-150 [&_svg]:hover:scale-110"
              >
                <User className="size-5" />
              </Link>
            ) : (
              <>
                <Link
                  href="/login"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Sign in
                </Link>
                <Link href="/signup" className={buttonVariants({ variant: "default", size: "sm" })}>
                  Sign up
                </Link>
              </>
            )}
          </nav>

          <button
            type="button"
            aria-label={isSearchOpen ? "Close search" : "Open search"}
            className="text-muted-foreground transition-colors hover:text-foreground lg:hidden"
            onClick={() => {
              setIsSearchOpen((open) => !open);
              setIsMenuOpen(false);
            }}
          >
            {isSearchOpen ? <X className="size-5" /> : <Search className="size-5" />}
          </button>
          <button
            type="button"
            aria-label={isMenuOpen ? "Close menu" : "Open menu"}
            className="text-muted-foreground transition-colors hover:text-foreground sm:hidden"
            onClick={() => {
              setIsMenuOpen((open) => !open);
              setIsSearchOpen(false);
            }}
          >
            {isMenuOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>

      {isSearchOpen && (
        <div className="animate-fade-in-down mx-auto mt-3 flex max-w-4xl justify-center px-4 lg:hidden">
          <SearchBar showSuggestions />
        </div>
      )}

      {isMenuOpen && (
        <nav className="animate-fade-in-down mx-auto mt-3 flex max-w-4xl flex-col gap-1 px-4 sm:hidden">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-lg px-2 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              {link.label}
            </Link>
          ))}
          {loggedIn ? (
            <Link
              href="/account"
              className="rounded-lg px-2 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              Account
            </Link>
          ) : (
            <>
              <Link
                href="/login"
                className="rounded-lg px-2 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              >
                Sign in
              </Link>
              <Link
                href="/signup"
                className={buttonVariants({ variant: "default", size: "sm", className: "mt-1 w-fit" })}
              >
                Sign up
              </Link>
            </>
          )}
        </nav>
      )}
    </header>
  );
}
