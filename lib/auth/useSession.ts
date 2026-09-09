"use client";

import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";

export interface SessionState {
  loggedIn: boolean;
  email: string | null;
  isLoading: boolean;
}

/**
 * Thin wrapper around the Supabase browser client's session state.
 * onAuthStateChange fires on sign-in/out/token-refresh in this tab and
 * across tabs (via localStorage), so every mounted consumer (e.g. the
 * header) picks up changes automatically — no custom event bus needed.
 */
export function useSession(): SessionState {
  const [state, setState] = useState<SessionState>({
    loggedIn: false,
    email: null,
    isLoading: true,
  });

  useEffect(() => {
    const supabase = createClient();

    supabase.auth.getSession().then(({ data: { session } }) => {
      setState({
        loggedIn: Boolean(session?.user),
        email: session?.user.email ?? null,
        isLoading: false,
      });
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setState({
        loggedIn: Boolean(session?.user),
        email: session?.user.email ?? null,
        isLoading: false,
      });
    });

    return () => subscription.unsubscribe();
  }, []);

  return state;
}
