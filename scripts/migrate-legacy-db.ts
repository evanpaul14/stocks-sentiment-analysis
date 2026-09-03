/**
 * One-off backfill from the old Flask app's SQLite databases into this
 * app's database. NOT run automatically by anything — invoke by hand when
 * you actually want the old data.
 *
 * The two schemas are NOT compatible (see CLAUDE.md: "ground-up rebuild,
 * no code was ported") — different file layout (old: two binds, ip_log.db
 * + blog.db; new: one file), different column names/types, no `slug` or
 * `image_*` columns on the old market_summary table, blog stored as HTML
 * rows in a DB table instead of MDX files. This script reads the old
 * files directly with better-sqlite3 (read-only) and re-inserts mapped
 * rows through this app's own Drizzle client, which self-migrates the
 * target DB first.
 *
 * Usage:
 *   npx tsx scripts/migrate-legacy-db.ts [--old-db <path>] [--old-blog-db <path>] [--apply]
 *
 * Defaults assume the old repo is a sibling directory:
 *   ../stocks-sentiment-analysis/instance/ip_log.db
 *   ../stocks-sentiment-analysis/instance/blog.db
 *
 * Without --apply this is a dry run: it reports row counts and a sample of
 * what would be written, but touches nothing. Every insert is a
 * conflict-safe upsert/ignore, so --apply is also safe to re-run.
 *
 * Known gaps (see comments at each section below):
 *   - article_images: migrated key-for-key (old cache key = sha256 of the
 *     article's link/title, same algorithm this app uses), but this app
 *     doesn't yet have a per-news-article image cache call site, so these
 *     rows won't be "hit" by anything until one exists. Harmless either
 *     way — worst case is an unused cache row.
 *   - blog_articles: old content is sanitized HTML, not Markdown/MDX. This
 *     script does a best-effort HTML->Markdown pass (see htmlToMarkdown
 *     below) since MDX can choke on unescaped/void HTML tags. Review the
 *     generated .mdx files before publishing — this is not a full HTML
 *     parser, just a few common-tag substitutions matching what the old
 *     WYSIWYG editor produced.
 *   - sentiment_page_cache: old separate intro/sentiment/prediction text
 *     columns get packed into this app's single `sections_json` blob
 *     ({ intro, sentimentSummary, prediction }).
 */

import path from "node:path";
import fs from "node:fs";
import Database from "better-sqlite3";
import { eq } from "drizzle-orm";

import { db, sqlite as targetSqlite } from "../lib/db/client";
import {
  marketWrap,
  marketWrapSendLog,
  articleImage,
  sentimentHistory,
  sentimentPageCache,
} from "../lib/db/schema";

interface CliArgs {
  oldDbPath: string;
  oldBlogDbPath: string;
  apply: boolean;
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  const get = (flag: string) => {
    const i = args.indexOf(flag);
    return i !== -1 ? args[i + 1] : undefined;
  };
  const oldRepoDefault = path.resolve(process.cwd(), "..", "stocks-sentiment-analysis");
  return {
    oldDbPath: get("--old-db") ?? path.join(oldRepoDefault, "instance", "ip_log.db"),
    oldBlogDbPath: get("--old-blog-db") ?? path.join(oldRepoDefault, "instance", "blog.db"),
    apply: args.includes("--apply"),
  };
}

/** Very small best-effort HTML->Markdown pass for the old WYSIWYG blog content.
 * Not a general HTML parser — handles the tag set the old editor actually
 * produced (p, headings, lists, links, bold/italic, br, img). Review output. */
function htmlToMarkdown(html: string): string {
  let text = html;
  text = text.replace(/<br\s*\/?>/gi, "\n");
  text = text.replace(/<\/p>\s*<p>/gi, "\n\n");
  text = text.replace(/<\/?p>/gi, "");
  text = text.replace(/<h1[^>]*>([\s\S]*?)<\/h1>/gi, "\n# $1\n");
  text = text.replace(/<h2[^>]*>([\s\S]*?)<\/h2>/gi, "\n## $1\n");
  text = text.replace(/<h3[^>]*>([\s\S]*?)<\/h3>/gi, "\n### $1\n");
  text = text.replace(/<strong[^>]*>([\s\S]*?)<\/strong>/gi, "**$1**");
  text = text.replace(/<b[^>]*>([\s\S]*?)<\/b>/gi, "**$1**");
  text = text.replace(/<em[^>]*>([\s\S]*?)<\/em>/gi, "*$1*");
  text = text.replace(/<i[^>]*>([\s\S]*?)<\/i>/gi, "*$1*");
  text = text.replace(/<a[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, "[$2]($1)");
  text = text.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, "- $1\n");
  text = text.replace(/<\/?(ul|ol)[^>]*>/gi, "\n");
  text = text.replace(/<img[^>]*src="([^"]*)"[^>]*>/gi, "![]($1)");
  text = text.replace(/<[^>]+>/g, ""); // drop anything else unrecognized
  text = text.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#39;/g, "'");
  return text.replace(/\n{3,}/g, "\n\n").trim();
}

function slugifyForFrontmatter(value: string): string {
  return value.trim();
}

function openOldDb(dbPath: string, label: string): Database.Database | null {
  if (!fs.existsSync(dbPath)) {
    console.warn(`[migrate] ${label} not found at ${dbPath} — skipping its tables`);
    return null;
  }
  return new Database(dbPath, { readonly: true, fileMustExist: true });
}

function migrateMarketSummary(oldDb: Database.Database, apply: boolean): Map<number, number> {
  const oldIdToNewId = new Map<number, number>();
  const rows = oldDb
    .prepare(
      `SELECT id, title, body, summary_date, index_snapshot, headline_sources, created_at
       FROM market_summary ORDER BY summary_date ASC`
    )
    .all() as Array<{
    id: number;
    title: string;
    body: string;
    summary_date: string;
    index_snapshot: string | null;
    headline_sources: string | null;
    created_at: string;
  }>;

  console.log(`[migrate] market_summary: ${rows.length} row(s) found in old db`);

  for (const row of rows) {
    const slug = row.summary_date; // old app's own slug convention is the ISO date, same as this app's
    console.log(`  - ${row.summary_date}: "${row.title}"${apply ? "" : " (dry run)"}`);

    if (!apply) continue;

    const inserted = db
      .insert(marketWrap)
      .values({
        date: row.summary_date,
        slug,
        title: row.title,
        body: row.body,
        indexSnapshotJson: row.index_snapshot ?? "[]",
        headlinesJson: row.headline_sources ?? "[]",
        createdAt: row.created_at,
      })
      .onConflictDoNothing({ target: marketWrap.date })
      .returning({ id: marketWrap.id })
      .get();

    const newId =
      inserted?.id ??
      db
        .select({ id: marketWrap.id })
        .from(marketWrap)
        .where(eq(marketWrap.date, row.summary_date))
        .get()?.id;

    if (newId != null) oldIdToNewId.set(row.id, newId);
  }

  return oldIdToNewId;
}

function migrateMarketSummarySendLog(
  oldDb: Database.Database,
  oldIdToNewId: Map<number, number>,
  apply: boolean
) {
  const rows = oldDb
    .prepare(
      `SELECT summary_id, sent_at, status, error_message FROM market_summary_send_log`
    )
    .all() as Array<{
    summary_id: number;
    sent_at: string;
    status: string;
    error_message: string | null;
  }>;

  console.log(`[migrate] market_summary_send_log: ${rows.length} row(s) found in old db`);

  let skippedNoParent = 0;
  for (const row of rows) {
    const newSummaryId = oldIdToNewId.get(row.summary_id);
    if (newSummaryId == null) {
      skippedNoParent++;
      continue;
    }
    if (!apply) continue;

    db.insert(marketWrapSendLog)
      .values({
        marketWrapId: newSummaryId,
        status: row.status === "failed" ? "failed" : "sent",
        errorMessage: row.error_message,
        sentAt: row.sent_at,
      })
      .onConflictDoNothing({ target: marketWrapSendLog.marketWrapId })
      .run();
  }
  if (skippedNoParent > 0) {
    console.log(
      `  - skipped ${skippedNoParent} send-log row(s) whose parent market_summary wasn't migrated (run with --apply, or check --old-db)`
    );
  }
}

function migrateArticleImages(oldDb: Database.Database, apply: boolean) {
  const rows = oldDb
    .prepare(
      `SELECT article_key, article_url, image_url, thumbnail_url,
              photographer_name, photographer_username, photographer_profile_url,
              unsplash_photo_id, created_at
       FROM article_images`
    )
    .all() as Array<{
    article_key: string;
    article_url: string | null;
    image_url: string;
    thumbnail_url: string | null;
    photographer_name: string | null;
    photographer_username: string | null;
    photographer_profile_url: string | null;
    unsplash_photo_id: string | null;
    created_at: string;
  }>;

  console.log(`[migrate] article_images: ${rows.length} row(s) found in old db`);
  if (!apply) return;

  const insert = db.insert(articleImage).values(
    rows.map((row) => ({
      key: row.article_key,
      unsplashId: row.unsplash_photo_id,
      url: row.image_url,
      thumbnailUrl: row.thumbnail_url,
      downloadLocation: null,
      attributionJson: JSON.stringify({
        photographerName: row.photographer_name,
        photographerUsername: row.photographer_username,
        photographerProfileUrl: row.photographer_profile_url,
      }),
      createdAt: row.created_at,
    }))
  );
  insert.onConflictDoNothing({ target: articleImage.key }).run();
}

function migrateSentimentHistory(oldDb: Database.Database, apply: boolean) {
  const rows = oldDb
    .prepare(
      `SELECT ticker, article_title, article_link, article_source,
              article_published_at, sentiment_label, analyzed_at
       FROM sentiment_history`
    )
    .all() as Array<{
    ticker: string;
    article_title: string;
    article_link: string | null;
    article_source: string | null;
    article_published_at: string | null;
    sentiment_label: string;
    analyzed_at: string;
  }>;

  console.log(`[migrate] sentiment_history: ${rows.length} row(s) found in old db`);
  if (!apply) return;

  const validLabels = new Set(["positive", "negative", "neutral"]);
  const valid = rows.filter((r) => validLabels.has(r.sentiment_label));
  if (valid.length !== rows.length) {
    console.log(`  - skipping ${rows.length - valid.length} row(s) with an unrecognized sentiment label`);
  }

  db.insert(sentimentHistory)
    .values(
      valid.map((row) => ({
        ticker: row.ticker,
        articleTitle: row.article_title,
        articleLink: row.article_link,
        articleSource: row.article_source,
        articlePublishedAt: row.article_published_at,
        sentiment: row.sentiment_label as "positive" | "negative" | "neutral",
        analyzedAt: row.analyzed_at,
      }))
    )
    .onConflictDoNothing({ target: [sentimentHistory.ticker, sentimentHistory.articleLink] })
    .run();
}

function migrateSentimentPageCache(oldDb: Database.Database, apply: boolean) {
  const rows = oldDb
    .prepare(
      `SELECT slug, ticker, intro_text, sentiment_text, prediction_text,
              price_data_json, sentiment_data_json, generated_at, expires_at
       FROM sentiment_page_cache`
    )
    .all() as Array<{
    slug: string;
    ticker: string;
    intro_text: string | null;
    sentiment_text: string | null;
    prediction_text: string | null;
    price_data_json: string | null;
    sentiment_data_json: string | null;
    generated_at: string;
    expires_at: string;
  }>;

  console.log(`[migrate] sentiment_page_cache: ${rows.length} row(s) found in old db`);

  for (const row of rows) {
    console.log(`  - ${row.slug} (${row.ticker})${apply ? "" : " (dry run)"}`);
    if (!apply) continue;

    db.insert(sentimentPageCache)
      .values({
        slug: row.slug,
        ticker: row.ticker,
        sectionsJson: JSON.stringify({
          intro: row.intro_text ?? "",
          sentimentSummary: row.sentiment_text ?? "",
          prediction: row.prediction_text ?? "",
        }),
        priceJson: row.price_data_json,
        sentimentJson: row.sentiment_data_json,
        generatedAt: row.generated_at,
        expiresAt: row.expires_at,
      })
      .onConflictDoUpdate({
        target: sentimentPageCache.slug,
        set: {
          ticker: row.ticker,
          sectionsJson: JSON.stringify({
            intro: row.intro_text ?? "",
            sentimentSummary: row.sentiment_text ?? "",
            prediction: row.prediction_text ?? "",
          }),
          priceJson: row.price_data_json,
          sentimentJson: row.sentiment_data_json,
          generatedAt: row.generated_at,
          expiresAt: row.expires_at,
        },
      })
      .run();
  }
}

function migrateBlogArticles(oldBlogDb: Database.Database, apply: boolean) {
  const rows = oldBlogDb
    .prepare(
      `SELECT title, slug, author, content, image_url, published_at
       FROM blog_articles WHERE is_published = 1`
    )
    .all() as Array<{
    title: string;
    slug: string;
    author: string;
    content: string;
    image_url: string | null;
    published_at: string | null;
  }>;

  console.log(`[migrate] blog_articles: ${rows.length} published row(s) found in old db`);
  if (rows.length === 0) return;

  const blogDir = path.join(process.cwd(), "content", "blog");
  for (const row of rows) {
    const slug = slugifyForFrontmatter(row.slug);
    const filePath = path.join(blogDir, `${slug}.mdx`);
    console.log(`  - ${slug}.mdx${apply ? "" : " (dry run)"}`);

    if (!apply) continue;
    if (fs.existsSync(filePath)) {
      console.log(`    already exists, skipping (delete it first to re-migrate)`);
      continue;
    }

    const publishedAt = (row.published_at ?? new Date().toISOString()).slice(0, 10);
    const body = htmlToMarkdown(row.content);
    const frontmatterLines = [
      "---",
      `title: ${JSON.stringify(row.title)}`,
      `description: ${JSON.stringify(body.slice(0, 160).replace(/\n/g, " "))}`,
      `author: ${JSON.stringify(row.author)}`,
      `publishedAt: ${JSON.stringify(publishedAt)}`,
      ...(row.image_url ? [`image: ${JSON.stringify(row.image_url)}`] : []),
      "tags: []",
      "---",
      "",
    ];
    fs.mkdirSync(blogDir, { recursive: true });
    fs.writeFileSync(filePath, frontmatterLines.join("\n") + body + "\n", "utf-8");
  }
  console.log(
    `  NOTE: review generated .mdx files for HTML->Markdown conversion artifacts before publishing.`
  );
}

async function main() {
  const { oldDbPath, oldBlogDbPath, apply } = parseArgs();

  console.log(`[migrate] mode: ${apply ? "APPLY (writing changes)" : "DRY RUN (pass --apply to write)"}`);
  console.log(`[migrate] old db: ${oldDbPath}`);
  console.log(`[migrate] old blog db: ${oldBlogDbPath}`);
  console.log("");

  const oldDb = openOldDb(oldDbPath, "old app db (ip_log.db)");
  if (oldDb) {
    const oldIdToNewId = migrateMarketSummary(oldDb, apply);
    migrateMarketSummarySendLog(oldDb, oldIdToNewId, apply);
    migrateArticleImages(oldDb, apply);
    migrateSentimentHistory(oldDb, apply);
    migrateSentimentPageCache(oldDb, apply);
    oldDb.close();
  }

  const oldBlogDb = openOldDb(oldBlogDbPath, "old blog db (blog.db)");
  if (oldBlogDb) {
    migrateBlogArticles(oldBlogDb, apply);
    oldBlogDb.close();
  }

  targetSqlite.close();
  console.log("");
  console.log(apply ? "[migrate] done." : "[migrate] dry run complete — re-run with --apply to write.");
}

main().catch((error) => {
  console.error("[migrate] failed", error);
  process.exitCode = 1;
});
