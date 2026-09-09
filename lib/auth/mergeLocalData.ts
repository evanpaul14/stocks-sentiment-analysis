import { readWatchlist, clearLocalWatchlist } from "@/lib/watchlist/storage";
import { readHistory, clearLocalHistory } from "@/lib/searchHistory/storage";

/**
 * One-time merge of whatever's in this device's localStorage into the
 * account right after a successful login/signup/OAuth callback, then clears
 * the local copies so the server is the unambiguous source of truth for
 * this (now logged-in) browser going forward.
 */
export async function mergeLocalDataIntoAccount(): Promise<void> {
  const watchlist = readWatchlist().map((e) => ({
    symbol: e.symbol,
    companyName: e.companyName,
  }));
  const searchHistory = readHistory();

  if (watchlist.length === 0 && searchHistory.length === 0) return;

  try {
    const response = await fetch("/api/account/merge-local-data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ watchlist, searchHistory }),
    });
    if (response.ok) {
      clearLocalWatchlist();
      clearLocalHistory();
    }
  } catch {
    // Network blip — local data just stays put and nothing was lost; the
    // next successful login/session refresh will retry the merge.
  }
}
