const SESSION_CHANGED_EVENT = "ssa-session-changed";

/** Call after login/signup/verify/logout so every mounted useSession() re-fetches. */
export function notifySessionChanged() {
  window.dispatchEvent(new Event(SESSION_CHANGED_EVENT));
}

export function subscribeToSessionChange(callback: () => void): () => void {
  window.addEventListener(SESSION_CHANGED_EVENT, callback);
  return () => window.removeEventListener(SESSION_CHANGED_EVENT, callback);
}
