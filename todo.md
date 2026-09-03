# TODO — things that need you

Everything in this list needs a human decision, real-world access, or money. Everything else
(all 7 phases of the rebuild) is built, typechecked, linted clean, and verified working against
live APIs in a production build.

## Blocking / broken right now

- **Gemini backup sentiment classifier is unusable — Google AI Studio billing credits are
  depleted.** The primary classifier (Cloudflare Workers AI) works fine, so sentiment analysis
  still works end-to-end today, but the backup path will fail (and correctly degrade to
  `"neutral"`) until you add credits at https://ai.studio/projects. Not a code problem — I hit
  this live while testing and confirmed the fallback behaves correctly either way.

## Should verify before relying on in production

- **Never live-tested the market-summary email subscribe flow**
  (`POST /api/market-summary/subscribe`) — it would send a real email through your live Mailgun
  account. Code is correct and typechecks; test it yourself once with a real address you control.
  Confirm `MAILGUN_DOMAIN` has its sending DNS records (SPF/DKIM) verified in Mailgun, or
  subscriber emails will bounce/land in spam.
- **`SITE_BASE_URL` in `.env` is set to `https://stocksentimentapp.com`** (carried over from the
  old app's config) — this feeds the sitemap, robots.txt, IndexNow, and all JSON-LD URLs. Update
  it if this rebuild is launching under a different domain, or leave as-is if it's the same site.
- **`ADMIN_API_TOKEN` and `INDEXNOW_KEY` were freshly generated this session** (random hex,
  stored in `.env`, gitignored). They're not shared anywhere unsafe, but treat them as live
  secrets — rotate them if you ever suspect exposure, same as any other credential in that file.

## Nothing committed to git yet

This project directory had no `.git` of its own when I started (a `git status` here was
resolving up to your home directory's repo by accident). I ran `git init` scoped correctly to
this project, but **I have not made any commits** — that's your call. When you're ready:

```bash
cd /Users/evanpaul/VSCodeProjects/stocks-sentiment-analysis-v2
git add .
git commit -m "Initial rebuild: Next.js stocks-sentiment-analysis-v2"
```

## Design polish (biggest gap vs. the original "fintech SaaS" ask)

The UI is fully functional and reasonably clean, but it's still using shadcn's **placeholder
neutral color palette**, not a distinctive brand identity. Everything works — search, charts,
sentiment streaming, trending, market summary, blog — but visually it reads as "well-built
generic dark UI," not yet the polished, distinctive fintech-SaaS look from the original brief.
If you want that final design pass, it's worth a dedicated session using the `frontend-design`
skill to:
- Pick a real color system (the current `--chart-1` through `--chart-5` tokens in
  `app/globals.css` are the ones to redefine)
- Add a real favicon/app icon (currently `app/manifest.ts` has an empty `icons: []`, and
  `public/` still has the default Next.js starter SVGs — `next.svg`, `vercel.svg`, etc. — which
  should be deleted once you have real brand assets)
- Give the particle background, charts, and cards a more considered visual treatment

## Deployment (not something I can do for you)

`DEPLOYMENT.md` has the full runbook (systemd service, Caddy/nginx reverse proxy, SQLite
backups via cron). None of it has been run against a real server — you'll need to:
1. Provision/point a VPS + domain
2. Copy the code over, fill in `.env` with production values
3. Follow `DEPLOYMENT.md` steps 3–6

## Nice-to-haves, not blockers

- **Blog has 2 starter posts** (`content/blog/*.mdx`) written by me as pipeline examples — add
  real content whenever.
- **Programmatic SEO sentiment pages cover 8 companies** (Magnificent 7 + Netflix, in
  `lib/utils/tickers.ts`). Add more by extending `SEO_SENTIMENT_COMPANIES`.
- **No automated test suite** — matches the old app (it had none either), but worth adding if
  you want CI to catch regressions going forward.
- **Analytics is off by default** — set `UMAMI_SCRIPT_URL` / `UMAMI_WEBSITE_ID` in `.env` if you
  stand up a self-hosted Umami instance (or swap the provider in
  `components/analytics/AnalyticsProvider.tsx`).
- **Dev machine note**: Node is running under Rosetta 2 (x86-64 build) on your Apple Silicon
  Mac — Next.js flags this as a dev/build performance drag. Installing an arm64 Node build would
  speed up local `npm run dev`/`npm run build`, purely a local convenience, not a deployment
  concern.
- **StockTwits has no official API** — the integration works today via a plain `fetch` with a
  realistic `User-Agent` (verified live, no Cloudflare challenge as of this build). If StockTwits
  tightens their bot protection later, `lib/integrations/stocktwits/fetchClient.ts` is the one
  place to add a workaround (see the comment there for what that'd involve).
