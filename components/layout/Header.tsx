"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, Search, X } from "lucide-react";
import { SearchBar } from "@/components/search/SearchBar";

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
    function handleScroll() {
      setIsScrolled(window.scrollY > 8);
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
      <div className="mx-auto flex max-w-4xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2 font-serif text-lg tracking-tight text-foreground">
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

        <nav className="hidden items-center gap-6 text-sm sm:flex">
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
          <button
            type="button"
            aria-label={isSearchOpen ? "Close search" : "Open search"}
            className="text-muted-foreground transition-colors hover:text-foreground"
            onClick={() => {
              setIsSearchOpen((open) => !open);
              setIsMenuOpen(false);
            }}
          >
            {isSearchOpen ? <X className="size-5" /> : <Search className="size-5" />}
          </button>
        </nav>

        <div className="flex items-center gap-4 sm:hidden">
          <button
            type="button"
            aria-label={isSearchOpen ? "Close search" : "Open search"}
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
        <div className="mx-auto mt-3 flex max-w-4xl justify-center px-4">
          <SearchBar showSuggestions />
        </div>
      )}

      {isMenuOpen && (
        <nav className="mx-auto mt-3 flex max-w-4xl flex-col gap-1 px-4 sm:hidden">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-lg px-2 py-2 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
