import type { NextRequest } from "next/server";
import { createReadOnlyClient } from "@/lib/supabase/server";

/** Resolves the Supabase user id for the current request's session, if any. */
export async function getCurrentUserId(request: NextRequest): Promise<string | null> {
  const user = await getCurrentUser(request);
  return user?.id ?? null;
}

/** Resolves the Supabase user id + email for the current request's session, if any. */
export async function getCurrentUser(
  request: NextRequest
): Promise<{ id: string; email: string | null } | null> {
  const supabase = createReadOnlyClient(request);
  const {
    data: { user },
  } = await supabase.auth.getUser();
  return user ? { id: user.id, email: user.email ?? null } : null;
}
