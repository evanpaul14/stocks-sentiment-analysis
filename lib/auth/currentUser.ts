import type { NextRequest } from "next/server";
import { createReadOnlyClient } from "@/lib/supabase/server";

/** Resolves the Supabase user id for the current request's session, if any. */
export async function getCurrentUserId(request: NextRequest): Promise<string | null> {
  const supabase = createReadOnlyClient(request);
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user?.id ?? null;
}
