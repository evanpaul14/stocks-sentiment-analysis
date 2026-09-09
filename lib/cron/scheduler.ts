import cron from "node-cron";
import { runMarketSummaryJob } from "./jobs/marketSummary";
import { runMag7SentimentJob } from "./jobs/mag7Backfill";
import { runPruneAuthDataJob } from "./jobs/pruneAuthData";
import { runOnceForDate } from "./lock";
import { isNyseTradingDay } from "@/lib/integrations/nyseCalendar";
import { currentEasternTime, todayInEastern } from "@/lib/utils/dates";
import * as marketSummary from "@/lib/db/queries/marketWrap";

const EASTERN_TZ = "America/New_York";

function isEnabled(name: string, defaultValue = true): boolean {
  const raw = process.env[name];
  if (raw == null) return defaultValue;
  return raw !== "0" && raw.toLowerCase() !== "false";
}

const RELEASE_HOUR = Number(process.env.MARKET_SUMMARY_RELEASE_HOUR ?? 16);
const RELEASE_MINUTE = Number(process.env.MARKET_SUMMARY_RELEASE_MINUTE ?? 15);

async function catchUpMarketSummaryIfNeeded() {
  const today = todayInEastern();
  if (!isNyseTradingDay(today)) return;

  const { hour, minute } = currentEasternTime();
  const pastReleaseTime =
    hour > RELEASE_HOUR || (hour === RELEASE_HOUR && minute >= RELEASE_MINUTE);
  if (!pastReleaseTime) return;

  const existing = await marketSummary.getByDate(today);
  if (existing) return;

  console.log(`[cron] catch-up: generating today's (${today}) market summary now`);
  await runOnceForDate("market-summary", today, runMarketSummaryJob);
}

let started = false;

/** Registers the daily market-summary and weekly MAG7 backfill cron jobs. */
export function startScheduler(): void {
  if (started) return;
  started = true;

  if (isEnabled("ENABLE_MARKET_SUMMARY")) {
    cron.schedule(
      `0 ${RELEASE_MINUTE} ${RELEASE_HOUR} * * 1-5`,
      async () => {
        const today = todayInEastern();
        await runOnceForDate("market-summary", today, runMarketSummaryJob);
      },
      { timezone: EASTERN_TZ }
    );

    catchUpMarketSummaryIfNeeded().catch((error) =>
      console.error("[cron] market summary catch-up failed", error)
    );
  }

  if (isEnabled("ENABLE_MAG7_SENTIMENT")) {
    cron.schedule(
      "0 0 6 * * 0",
      async () => {
        const today = todayInEastern();
        await runOnceForDate("mag7-backfill", today, runMag7SentimentJob);
      },
      { timezone: EASTERN_TZ }
    );
  }

  cron.schedule(
    "0 30 3 * * *",
    async () => {
      const today = todayInEastern();
      await runOnceForDate("prune-auth-data", today, runPruneAuthDataJob);
    },
    { timezone: EASTERN_TZ }
  );

  console.log("[cron] scheduler started");
}
