"use client";

import { useCallback, useEffect, useState } from "react";
import { subscribeToSessionChange } from "./sessionEvents";

export interface SessionState {
  loggedIn: boolean;
  email: string | null;
  isLoading: boolean;
}

function readLoggedInFlag(): boolean {
  if (typeof document === "undefined") return false;
  return document.cookie
    .split("; ")
    .some((c) => c === "ssa_logged_in=1");
}

/**
 * Fast path: the non-httpOnly `ssa_logged_in` flag cookie renders the
 * logged-in/out shell instantly with no request. Source of truth: one fetch
 * to /api/account/session reconciles it (and catches a session that expired
 * server-side without the client noticing).
 */
export function useSession(): SessionState {
  const [state, setState] = useState<SessionState>(() => ({
    loggedIn: readLoggedInFlag(),
    email: null,
    isLoading: true,
  }));

  const refetch = useCallback(() => {
    let cancelled = false;
    fetch("/api/account/session")
      .then((res) => (res.ok ? res.json() : { loggedIn: false, email: null }))
      .then((data) => {
        if (cancelled) return;
        setState({ loggedIn: Boolean(data.loggedIn), email: data.email ?? null, isLoading: false });
      })
      .catch(() => {
        if (!cancelled) setState((s) => ({ ...s, isLoading: false }));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => refetch(), [refetch]);

  // Re-fetch on login/signup/verify/logout so already-mounted consumers
  // (e.g. the header) pick up the change without a full page reload.
  useEffect(() => subscribeToSessionChange(refetch), [refetch]);

  return state;
}
