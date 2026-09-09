import { createClient as createSupabaseClient } from "@supabase/supabase-js";

/**
 * Service-role client — bypasses RLS and can manage any user. Server-only;
 * SUPABASE_SERVICE_ROLE_KEY must never reach the browser. Used only for
 * account deletion (supabase.auth.admin.deleteUser), which the regular
 * client SDK can't do for a user's own account.
 */
export function createAdminClient() {
  return createSupabaseClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    { auth: { autoRefreshToken: false, persistSession: false } }
  );
}
