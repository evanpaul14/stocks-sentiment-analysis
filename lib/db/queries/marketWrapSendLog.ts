import { eq } from "drizzle-orm";
import { db } from "../client";
import { marketWrapSendLog } from "../schema";

export async function hasSentSuccessfully(marketWrapId: number) {
  const row = await db.query.marketWrapSendLog.findFirst({
    where: eq(marketWrapSendLog.marketWrapId, marketWrapId),
  });
  return row?.status === "sent";
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
