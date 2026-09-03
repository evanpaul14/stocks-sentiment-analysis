import type { Metadata } from "next";
import Link from "next/link";
import { getAllBlogPosts } from "@/lib/blog/posts";
import { SEO_SENTIMENT_COMPANIES, companySlug } from "@/lib/utils/tickers";

export const metadata: Metadata = {
  title: "Blog — Stock Sentiment",
  description: "Product updates and explainers from the Stock Sentiment team.",
  alternates: { canonical: "/blog" },
};

export default function BlogIndexPage() {
  const posts = getAllBlogPosts();

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-6 text-2xl font-semibold">Blog</h1>

      <section className="mb-10">
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">Stock sentiment reports</h2>
        <ul className="flex flex-wrap gap-2">
          {SEO_SENTIMENT_COMPANIES.map((company) => (
            <li key={company.ticker}>
              <Link
                href={`/blog/${companySlug(company.companyName)}`}
                className="block rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-ring hover:text-foreground"
              >
                {company.companyName} ({company.ticker})
              </Link>
            </li>
          ))}
        </ul>
      </section>

      {posts.length === 0 ? (
        <p className="text-sm text-muted-foreground">No posts published yet.</p>
      ) : (
        <ul className="space-y-6">
          {posts.map((post) => (
            <li key={post.slug}>
              <Link href={`/blog/${post.slug}`} className="block hover:opacity-80">
                <h2 className="text-lg font-medium">{post.frontmatter.title}</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {post.frontmatter.description}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {new Date(post.frontmatter.publishedAt).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                  {" · "}
                  {post.frontmatter.author}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
