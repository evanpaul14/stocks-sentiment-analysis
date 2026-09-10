-- Separates the stable first-publish timestamp from the rolling 24h
-- cache-refresh timestamp (previously both were the same `generated_at`
-- column, which got overwritten on every cache refresh) — see
-- lib/blog/seoSentimentPageData.ts and the Article schema's
-- datePublished/dateModified split.
ALTER TABLE sentiment_page_cache ADD COLUMN first_generated_at TEXT;

UPDATE sentiment_page_cache SET first_generated_at = generated_at
WHERE first_generated_at IS NULL;
