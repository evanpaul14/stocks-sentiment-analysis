import { isNotNull, lt, or, sql } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { authToken, session } from "@/lib/db/schema";
import * as sessions from "@/lib/db/queries/sessions";

/** Daily sweep: rows that resolveSession/getValidByHash already treat as dead, just tidying up storage. */
export async function runPruneAuthDataJob() {
  sessions.deleteExpired();
  db.delete(session).where(isNotNull(session.revokedAt)).run();

  db.delete(authToken)
    .where(or(lt(authToken.expiresAt, sql`(current_timestamp)`), isNotNull(authToken.consumedAt)))
    .run();
}
