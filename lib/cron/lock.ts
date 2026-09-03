import { tryClaim } from "@/lib/db/queries/jobRunLog";

/** Runs fn only if this process wins the (job, date) claim — survives restarts via SQLite. */
export async function runOnceForDate(
  jobName: string,
  runDate: string,
  fn: () => Promise<void>
): Promise<void> {
  if (!tryClaim(jobName, runDate)) {
    console.log(`[cron] ${jobName} already claimed for ${runDate}, skipping`);
    return;
  }
  await fn();
}
