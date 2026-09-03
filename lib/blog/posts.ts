import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";

const BLOG_DIR = path.join(process.cwd(), "content", "blog");
const DEFAULT_AUTHOR = process.env.BLOG_DEFAULT_AUTHOR ?? "Stock Sentiment Team";

export interface BlogPostFrontmatter {
  title: string;
  description: string;
  author: string;
  publishedAt: string;
  image?: string;
  tags?: string[];
}

export interface BlogPost {
  slug: string;
  frontmatter: BlogPostFrontmatter;
  content: string;
}

function readAllSlugs(): string[] {
  if (!fs.existsSync(BLOG_DIR)) return [];
  return fs
    .readdirSync(BLOG_DIR)
    .filter((f) => f.endsWith(".mdx"))
    .map((f) => f.replace(/\.mdx$/, ""));
}

function readPost(slug: string): BlogPost | null {
  const filePath = path.join(BLOG_DIR, `${slug}.mdx`);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, "utf-8");
  const { data, content } = matter(raw);

  return {
    slug,
    frontmatter: {
      title: data.title ?? slug,
      description: data.description ?? "",
      author: data.author ?? DEFAULT_AUTHOR,
      publishedAt: data.publishedAt ?? new Date().toISOString(),
      image: data.image,
      tags: data.tags,
    },
    content,
  };
}

/** All blog posts (from content/blog/*.mdx), newest first. */
export function getAllBlogPosts(): BlogPost[] {
  return readAllSlugs()
    .map(readPost)
    .filter((p): p is BlogPost => p !== null)
    .sort((a, b) => (a.frontmatter.publishedAt < b.frontmatter.publishedAt ? 1 : -1));
}

export function getBlogPostBySlug(slug: string): BlogPost | null {
  return readPost(slug);
}

export function getAllBlogSlugs(): string[] {
  return readAllSlugs();
}
