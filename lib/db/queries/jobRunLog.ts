import { sqlite } from "../client";

/**
 * Claims a (job, date) slot. Returns true if this call won the claim (should run),
 * false if another run already claimed it today. Survives process restarts,
 * unlike an in-memory flag — this is the cron dedupe lock.
 */
export function tryClaim(jobName: string, runDate: string): boolean {
  const result = sqlite
    .prepare(
      "INSERT OR IGNORE INTO job_run_log (job_name, run_date) VALUES (?, ?)"
    )
    .run(jobName, runDate);
  return result.changes > 0;
}
