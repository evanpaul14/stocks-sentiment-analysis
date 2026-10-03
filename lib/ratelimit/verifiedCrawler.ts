import { promises as dns } from "node:dns";
import { BlockList, isIP } from "node:net";

/**
 * Crawlers share a handful of IPs and would burn through the anonymous
 * search quota instantly, de-indexing every stock page. A User-Agent alone
 * is spoofable, so a claimed bot is only exempt if its IP checks out:
 *  - `rdns`: reverse DNS must land under one of these suffixes AND that
 *    hostname must resolve forward to the same IP (the verification Google,
 *    Bing and Apple document).
 *  - `ranges`: IP must be inside the operator's published CIDR list (how
 *    DuckDuckGo, OpenAI, Perplexity and Anthropic do it — they have no rDNS).
 */
interface Bot {
  ua: RegExp;
  rdns?: string[];
  ranges?: string;
}

const BOTS: Bot[] = [
  {
    ua: /googlebot|google-inspectiontool|googleother|storebot-google/i,
    rdns: [".googlebot.com", ".google.com", ".googleusercontent.com"],
  },
  { ua: /bingbot|bingpreview|msnbot/i, rdns: [".search.msn.com"] },
  { ua: /applebot/i, rdns: [".applebot.apple.com"] },
  { ua: /duckduckbot/i, ranges: "https://duckduckgo.com/duckduckbot.json" },
  { ua: /gptbot/i, ranges: "https://openai.com/gptbot.json" },
  { ua: /oai-searchbot/i, ranges: "https://openai.com/searchbot.json" },
  { ua: /chatgpt-user/i, ranges: "https://openai.com/chatgpt-user.json" },
  { ua: /perplexitybot/i, ranges: "https://www.perplexity.com/perplexitybot.json" },
  { ua: /perplexity-user/i, ranges: "https://www.perplexity.com/perplexity-user.json" },
  { ua: /claudebot|claude-user|claude-searchbot/i, ranges: "https://claude.com/crawling/bots.json" },
];

const RANGES_TTL_MS = 24 * 60 * 60 * 1000;
const RANGES_RETRY_MS = 10 * 60 * 1000;
const POSITIVE_TTL_MS = 24 * 60 * 60 * 1000;
const NEGATIVE_TTL_MS = 60 * 60 * 1000;
const MAX_VERDICTS = 5000;

// Keyed by `${ip}|${bot index}`. Bounded so spoofed-UA traffic can't grow it forever.
const verdicts = new Map<string, { ok: boolean; expires: number }>();

interface RangeEntry {
  list: BlockList | null;
  expires: number;
  loading: Promise<void> | null;
}
const rangeCache = new Map<string, RangeEntry>();

function normalizeIp(ip: string): string {
  return ip.replace(/^::ffff:/i, "");
}

function ipFamily(ip: string): "ipv4" | "ipv6" | null {
  const v = isIP(ip);
  return v === 4 ? "ipv4" : v === 6 ? "ipv6" : null;
}

async function loadRanges(url: string, entry: RangeEntry) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = (await res.json()) as {
      prefixes?: { ipv4Prefix?: string; ipv6Prefix?: string }[];
    };
    const list = new BlockList();
    for (const p of data.prefixes ?? []) {
      const prefix = p.ipv4Prefix ?? p.ipv6Prefix;
      if (!prefix) continue;
      const [addr, bits] = prefix.split("/");
      list.addSubnet(addr, Number(bits ?? (p.ipv4Prefix ? 32 : 128)), p.ipv4Prefix ? "ipv4" : "ipv6");
    }
    entry.list = list;
    entry.expires = Date.now() + RANGES_TTL_MS;
  } catch (error) {
    // Keep serving the stale list if we have one; otherwise fail closed.
    console.error(`[crawler-verify] failed to load ${url}:`, error);
    entry.expires = Date.now() + RANGES_RETRY_MS;
  }
}

async function inPublishedRanges(url: string, ip: string): Promise<boolean> {
  let entry = rangeCache.get(url);
  if (!entry) {
    entry = { list: null, expires: 0, loading: null };
    rangeCache.set(url, entry);
  }
  if (Date.now() >= entry.expires) {
    const e = entry;
    e.loading ??= loadRanges(url, e).finally(() => {
      e.loading = null;
    });
    // Block only when there's nothing to check against yet.
    if (!e.list) await e.loading;
  }
  const family = ipFamily(ip);
  return Boolean(entry.list && family && entry.list.check(ip, family));
}

async function reverseForwardMatches(ip: string, suffixes: string[]): Promise<boolean> {
  const family = ipFamily(ip);
  if (!family) return false;
  const self = new BlockList();
  self.addAddress(ip, family);

  let hostnames: string[];
  try {
    hostnames = await dns.reverse(ip);
  } catch {
    return false;
  }
  for (const host of hostnames) {
    const lower = host.toLowerCase().replace(/\.$/, "");
    if (!suffixes.some((s) => lower.endsWith(s))) continue;
    const forward = await Promise.all([
      dns.resolve4(lower).catch(() => [] as string[]),
      dns.resolve6(lower).catch(() => [] as string[]),
    ]);
    for (const addr of forward.flat()) {
      const f = ipFamily(addr);
      if (f && self.check(addr, f)) return true;
    }
  }
  return false;
}

/** True only if the UA claims to be a known crawler AND the IP proves it. */
export async function isVerifiedCrawler(ip: string, userAgent: string | null): Promise<boolean> {
  if (!userAgent) return false;
  const botIndex = BOTS.findIndex((b) => b.ua.test(userAgent));
  if (botIndex === -1) return false;

  const clean = normalizeIp(ip);
  const key = `${clean}|${botIndex}`;
  const cached = verdicts.get(key);
  if (cached && cached.expires > Date.now()) return cached.ok;

  const bot = BOTS[botIndex];
  let ok = false;
  if (bot.rdns) ok = await reverseForwardMatches(clean, bot.rdns);
  if (!ok && bot.ranges) ok = await inPublishedRanges(bot.ranges, clean);

  if (verdicts.size >= MAX_VERDICTS) verdicts.clear();
  verdicts.set(key, { ok, expires: Date.now() + (ok ? POSITIVE_TTL_MS : NEGATIVE_TTL_MS) });
  return ok;
}
