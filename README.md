# Stock Sentiment

A modern rebuild of the stocks-sentiment-analysis app — real-time stock prices, historical
charts, AI-powered news sentiment analysis, trending dashboards, an automated daily market
summary, and a blog — built as a single Next.js app (App Router, TypeScript), self-hosted on a
VPS with SQLite for storage.

This is a ground-up rewrite of the original Flask app. Only the feature set and the external API
integrations were carried over; none of the original code was reused.

## Stack

- **Next.js 16** (App Router, TypeScript, Turbopack)
- **SQLite** via `better-sqlite3` + a typed Drizzle query layer — no external database
- **Tailwind CSS + shadcn/ui** (`base-nova` style)
- **Recharts** for charts
- **node-cron** for the daily market-summary and weekly sentiment-backfill jobs, running
  in-process (see `instrumentation.ts` / `lib/cron/`)

## Quickstart

```bash
npm install
cp .env.example .env   # fill in real API keys — see comments in the file
npm run dev
```

Open http://localhost:3000.

## Environment variables

See `.env.example` for the full list with comments. Everything is optional except
`GOOGLE_API_KEY` (Gemini backup sentiment classifier) — every other integration degrades
gracefully (returns empty results, or skips the feature) if its keys are unset.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Start the dev server |
| `npm run build` | Production build (also runs `prebuild`, which writes the IndexNow verification file) |
| `npm run start` | Run the production build |
| `npm run db:migrate` | Apply any pending SQLite migrations (also happens automatically on first DB connection) |
| `npm run lint` | ESLint |

## Project structure

```
app/                    Pages and API routes (App Router)
lib/
  db/                    SQLite client, schema (Drizzle), migrations, per-table query modules
  integrations/          One module per external API (Yahoo, StockTwits, Mailgun, etc.)
  cron/                  node-cron scheduler + the two scheduled jobs
  cache/                 In-memory TTL cache for hot short-lived data
  ratelimit/              In-memory per-IP token bucket
  auth/                  Single admin bearer-token check (no accounts system)
  seo/                   Structured data (JSON-LD), CSP nonce helper, sitemap key file writer
content/blog/*.mdx       Blog posts — see "Publishing a blog post" below
components/              UI, organized by feature area
```

**Adding a new data source**: put typed functions in `lib/integrations/<source>/*.ts` (no
Next.js imports there), add a cache wrapper in `lib/cache` if it needs one, expose it via one
thin `app/api/.../route.ts`, and call it from a page. Route handlers stay compose-only — that's
the pattern the whole app follows.

## Publishing a blog post

There's no admin UI — posts are `.mdx` files with frontmatter:

```mdx
---
title: "Post Title"
description: "One-sentence summary for SEO/social."
author: "Your Name"
publishedAt: "2026-08-22"
tags: ["announcements"]
---

Post body in Markdown/MDX goes here.
```

Drop it in `content/blog/`, commit, redeploy. It'll appear at `/blog/<filename-without-extension>`.

## Deployment

Runs as a systemd service on a self-hosted VPS behind Caddy, with SQLite backed up via
`deploy/backup-db.sh`. `.github/workflows/deploy.yml` builds and deploys automatically on every
push to `master`.

## What's left to do

See [`todo.md`](./todo.md) for a running list of things that need a human (API credentials to
verify, DNS/mail setup, design polish opportunities, etc).
