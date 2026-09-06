import { db, sqlite } from "../client";
import { marketWrapSendLog } from "../schema";

/**
 * Atomically claims the right to send the email for marketWrapId. Only one
 * caller — even across concurrent processes/requests, e.g. the cron job
 * racing an admin-triggered regenerate — wins the claim; everyone else
 * (already sent, or another send is mid-flight) gets false and must not send.
 * Winning the claim leaves the row as 'pending'; the caller must follow up
 * with recordSend to finalize it to 'sent' or 'failed'.
 */
export function tryClaimSend(marketWrapId: number): boolean {
  const result = sqlite
    .prepare(
      `INSERT INTO market_wrap_send_log (market_wrap_id, status)
       VALUES (?, 'pending')
       ON CONFLICT(market_wrap_id) DO UPDATE SET status = 'pending'
       WHERE market_wrap_send_log.status = 'failed'`
    )
    .run(marketWrapId);
  return result.changes > 0;
}

export function recordSend(
  marketWrapId: number,
  status: "sent" | "failed",
  errorMessage?: string
) {
  return db
    .insert(marketWrapSendLog)
    .values({ marketWrapId, status, errorMessage: errorMessage ?? null })
    .onConflictDoUpdate({
      target: marketWrapSendLog.marketWrapId,
      set: { status, errorMessage: errorMessage ?? null },
    })
    .returning()
    .get();
}
