const HISTORY_KEY = "ssa_recent_searches_v1";
const HISTORY_CHANGED_EVENT = "ssa-search-history-changed";
const MAX_HISTORY = 8;

let cachedRaw: string | null | undefined;
let cachedHistory: string[] = [];

export function readHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    // Return the same array reference when the underlying value hasn't
    // changed — required for useSyncExternalStore's getSnapshot to avoid
    // re-rendering (and looping) on every call.
    if (raw === cachedRaw) return cachedHistory;

    cachedRaw = raw;
    if (!raw) {
      cachedHistory = [];
    } else {
      const parsed = JSON.parse(raw);
      cachedHistory = Array.isArray(parsed)
        ? parsed.filter((x) => typeof x === "string")
        : [];
    }
    return cachedHistory;
  } catch {
    return [];
  }
}

export function writeHistory(entries: string[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
    window.dispatchEvent(new Event(HISTORY_CHANGED_EVENT));
  } catch {
    // localStorage unavailable (private browsing, etc.) — degrade silently
  }
}

export function recordSearch(term: string) {
  const trimmed = term.trim();
  if (!trimmed) return;
  const current = readHistory();
  writeHistory([trimmed, ...current.filter((h) => h.toLowerCase() !== trimmed.toLowerCase())]);
}

export function clearLocalHistory() {
  try {
    localStorage.removeItem(HISTORY_KEY);
    window.dispatchEvent(new Event(HISTORY_CHANGED_EVENT));
  } catch {
    // localStorage unavailable — degrade silently
  }
}

export function subscribeToHistory(callback: () => void): () => void {
  window.addEventListener(HISTORY_CHANGED_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(HISTORY_CHANGED_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}
