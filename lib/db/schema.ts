import { sql } from "drizzle-orm";
import {
  index,
  integer,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

/** Unsplash thumbnail cache, keyed by a hash of the article link/title. */
export const articleImage = sqliteTable("article_image", {
  key: text("key").primaryKey(),
  unsplashId: text("unsplash_id"),
  url: text("url").notNull(),
  thumbnailUrl: text("thumbnail_url"),
  downloadLocation: text("download_location"),
  attributionJson: text("attribution_json"),
  createdAt: text("created_at")
    .notNull()
    .default(sql`(current_timestamp)`),
});

/** One row per trading-day market wrap. */
export const marketWrap = sqliteTable(
  "market_wrap",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    date: text("date").notNull(),
    slug: text("slug").notNull(),
    title: text("title").notNull(),
    body: text("body").notNull(),
    indexSnapshotJson: text("index_snapshot_json").notNull(),
    headlinesJson: text("headlines_json").notNull(),
    imageUrl: text("image_url"),
    imageThumbnailUrl: text("image_thumbnail_url"),
    imagePhotographerName: text("image_photographer_name"),
    imagePhotographerProfileUrl: text("image_photographer_profile_url"),
    createdAt: text("created_at")
      .notNull()
      .default(sql`(current_timestamp)`),
  },
  (table) => [
    uniqueIndex("market_wrap_date_idx").on(table.date),
    uniqueIndex("market_wrap_slug_idx").on(table.slug),
  ]
);

/** Dedupes the daily Mailgun broadcast — one send per market_wrap. */
export const marketWrapSendLog = sqliteTable(
  "market_wrap_send_log",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    marketWrapId: integer("market_wrap_id")
      .notNull()
      .references(() => marketWrap.id),
    status: text("status", { enum: ["sent", "failed"] }).notNull(),
    errorMessage: text("error_message"),
    sentAt: text("sent_at")
      .notNull()
      .default(sql`(current_timestamp)`),
  },
  (table) => [
    uniqueIndex("market_wrap_send_log_summary_idx").on(table.marketWrapId),
  ]
);

/** Per-article sentiment classification results. */
export const sentimentHistory = sqliteTable(
  "sentiment_history",
  {
    id: integer("id").primaryKey({ autoIncrement: true }),
    ticker: text("ticker").notNull(),
    articleTitle: text("article_title").notNull(),
    articleLink: text("article_link"),
    articleSource: text("article_source"),
    articlePublishedAt: text("article_published_at"),
    sentiment: text("sentiment", {
      enum: ["positive", "negative", "neutral"],
    }).notNull(),
    analyzedAt: text("analyzed_at")
      .notNull()
      .default(sql`(current_timestamp)`),
  },
  (table) => [
    uniqueIndex("sentiment_history_ticker_link_idx").on(
      table.ticker,
      table.articleLink
    ),
    index("sentiment_history_ticker_date_idx").on(
      table.ticker,
      table.analyzedAt
    ),
  ]
);

/** TTL cache for programmatic SEO "sentiment of X stock" pages. */
export const sentimentPageCache = sqliteTable("sentiment_page_cache", {
  slug: text("slug").primaryKey(),
  ticker: text("ticker").notNull(),
  sectionsJson: text("sections_json").notNull(),
  priceJson: text("price_json"),
  sentimentJson: text("sentiment_json"),
  generatedAt: text("generated_at")
    .notNull()
    .default(sql`(current_timestamp)`),
  expiresAt: text("expires_at").notNull(),
});

/** Dedupe lock for cron jobs — survives process restarts, unlike an in-memory flag. */
export const jobRunLog = sqliteTable(
  "job_run_log",
  {
    jobName: text("job_name").notNull(),
    runDate: text("run_date").notNull(),
    ranAt: text("ran_at")
      .notNull()
      .default(sql`(current_timestamp)`),
  },
  (table) => [
    uniqueIndex("job_run_log_job_date_idx").on(table.jobName, table.runDate),
  ]
);
