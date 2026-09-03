import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { MDXRemote } from "next-mdx-remote/rsc";
import { getAllBlogPosts, getAllBlogSlugs, getBlogPostBySlug } from "@/lib/blog/posts";
import { getSeoSentimentPageData } from "@/lib/blog/seoSentimentPageData";
import { companySlug } from "@/lib/utils/tickers";
import { SentimentPriceOverlayChart } from "@/components/blog/SentimentPriceOverlayChart";
import { JsonLd } from "@/lib/seo/JsonLd";
import { articleJsonLd } from "@/lib/seo/structuredData";

interface BlogSlugPageProps {
  params: Promise<{ slug: string }>;
}

// Only pre-render real blog posts; programmatic SEO sentiment pages (whose
// slugs also live at /blog/[slug]) render on-demand and are cached in
// sentiment_page_cache instead — see getSeoSentimentPageData.
export function generateStaticParams() {
  return getAllBlogSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: BlogSlugPageProps): Promise<Metadata> {
  const { slug } = await params;

  const post = getBlogPostBySlug(slug);
  if (post) {
    return {
      title: post.frontmatter.title,
      description: post.frontmatter.description,
      alternates: { canonical: `/blog/${slug}` },
      openGraph: {
        title: post.frontmatter.title,
        description: post.frontmatter.description,
        type: "article",
        publishedTime: post.frontmatter.publishedAt,
        url: `/blog/${slug}`,
      },
      twitter: {
        card: "summary_large_image",
        title: post.frontmatter.title,
        description: post.frontmatter.description,
      },
    };
  }

  const seoPage = await getSeoSentimentPageData(slug);
  if (seoPage) {
    const title = `${seoPage.company.companyName} (${seoPage.company.ticker}) Stock Sentiment`;
    return {
      title,
      description: seoPage.sections.intro,
      alternates: { canonical: `/blog/${slug}` },
      openGraph: {
        title,
        description: seoPage.sections.intro,
        type: "article",
        url: `/blog/${slug}`,
      },
      twitter: {
        card: "summary_large_image",
        title,
        description: seoPage.sections.intro,
      },
    };
  }

  return {};
}

export default async function BlogSlugPage({ params }: BlogSlugPageProps) {
  const { slug } = await params;

  const post = getBlogPostBySlug(slug);
  if (post) return <BlogPostView slug={slug} />;

  const seoPage = await getSeoSentimentPageData(slug);
  if (seoPage) return <SeoSentimentPageView data={seoPage} />;

  notFound();
}

function BlogPostView({ slug }: { slug: string }) {
  const post = getBlogPostBySlug(slug)!;
  const related = getAllBlogPosts()
    .filter((p) => p.slug !== slug)
    .slice(0, 3);

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <JsonLd
        data={articleJsonLd({
          headline: post.frontmatter.title,
          description: post.frontmatter.description,
          datePublished: post.frontmatter.publishedAt,
          url: `${process.env.SITE_BASE_URL ?? ""}/blog/${slug}`,
          author: post.frontmatter.author,
        })}
      />
      <article>
        <h1 className="text-2xl font-semibold">{post.frontmatter.title}</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          {new Date(post.frontmatter.publishedAt).toLocaleDateString("en-US", {
            year: "numeric",
            month: "long",
            day: "numeric",
          })}
          {" · "}
          {post.frontmatter.author}
        </p>

        <div className="prose prose-invert mt-6 max-w-none text-sm leading-relaxed">
          <MDXRemote source={post.content} />
        </div>
      </article>

      {related.length > 0 && (
        <section className="mt-12 border-t border-border pt-6">
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">More posts</h2>
          <ul className="space-y-1">
            {related.map((p) => (
              <li key={p.slug}>
                <Link href={`/blog/${p.slug}`} className="text-sm hover:underline">
                  {p.frontmatter.title}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function SeoSentimentPageView({
  data,
}: {
  data: NonNullable<Awaited<ReturnType<typeof getSeoSentimentPageData>>>;
}) {
  const { company, sections, overlay, related } = data;

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <JsonLd
        data={articleJsonLd({
          headline: `${company.companyName} (${company.ticker}) Stock Sentiment`,
          description: sections.intro,
          datePublished: data.generatedAt,
          url: `${process.env.SITE_BASE_URL ?? ""}/blog/${companySlug(company.companyName)}`,
        })}
      />
      <h1 className="text-2xl font-semibold">
        {company.companyName} ({company.ticker}) Stock Sentiment
      </h1>

      <div className="prose prose-invert mt-4 max-w-none text-sm leading-relaxed">
        <p>{sections.intro}</p>
        <h2 className="text-lg font-medium">Recent Sentiment</h2>
        <p>{sections.sentimentSummary}</p>
      </div>

      <section className="mt-6 rounded-xl border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-medium text-muted-foreground">
          Sentiment vs. Price (90 days)
        </h2>
        <SentimentPriceOverlayChart data={overlay} />
      </section>

      <div className="prose prose-invert mt-6 max-w-none text-sm leading-relaxed">
        <h2 className="text-lg font-medium">Outlook</h2>
        <p>{sections.prediction}</p>
      </div>

      <p className="mt-4">
        <Link href={`/stock/${company.ticker}`} className="text-sm hover:underline">
          View live {company.ticker} price and news →
        </Link>
      </p>

      {related.length > 0 && (
        <section className="mt-10 border-t border-border pt-6">
          <h2 className="mb-3 text-sm font-medium text-muted-foreground">Related companies</h2>
          <ul className="flex flex-wrap gap-2">
            {related.map((c) => (
              <li key={c.ticker}>
                <Link
                  href={`/blog/${companySlug(c.companyName)}`}
                  className="rounded-full border border-border px-3 py-1 text-xs hover:bg-muted"
                >
                  {c.companyName}
                </Link>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
