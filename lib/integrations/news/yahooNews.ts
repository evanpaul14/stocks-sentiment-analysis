import { yahooFinance } from "../yahoo/client";
import type { NewsArticle } from "./types";

const CORPORATE_SUFFIX =
  /[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings|group|class [a-z])\.?$/gi;

/** Extra names a company's headlines commonly use instead of its legal name. */
const NAME_ALIASES: Record<string, string[]> = {
  alphabet: ["google"],
  meta: ["facebook"],
};

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Yahoo's own quote/landing pages show up in news results; they aren't articles. */
const QUOTE_PAGE_TITLE = /stock price,? (news,? )?quote|quote (&|and) history/i;

/**
 * Yahoo tags nearly every result with the searched ticker (`relatedTickers`),
 * so the tag is useless as a relevance signal; headlines have to actually name
 * the ticker or company.
 */
function buildTitleMatcher(
  query: string,
  companyName: string | undefined,
): RegExp {
  const terms = new Set<string>([query]);
  if (companyName) {
    const short = companyName.replace(CORPORATE_SUFFIX, "").trim();
    if (short.length >= 3) {
      terms.add(short);
      for (const alias of NAME_ALIASES[short.toLowerCase()] ?? [])
        terms.add(alias);
    }
  }
  const alternation = [...terms].map(escapeRegex).join("|");
  return new RegExp(`(?<![A-Za-z0-9])(${alternation})(?![A-Za-z0-9])`, "i");
}

/**
 * Fallback news from Yahoo's search endpoint, filtered to headlines that name
 * the ticker/company. `query` is a ticker or free-text company name. One
 * search call returns both the matching quote (for the company name) and news.
 */
export async function fetchYahooNews(
  query: string,
  limit: number,
): Promise<NewsArticle[]> {
  const result = await yahooFinance.search(query, {
    newsCount: 50,
    quotesCount: 1,
  });
  const quote = result.quotes.find((q) => "symbol" in q);
  const rawName = quote && (quote.longname ?? quote.shortname);
  const companyName = typeof rawName === "string" ? rawName : undefined;
  const matcher = buildTitleMatcher(query, companyName);

  return (result.news ?? [])
    .filter(
      (item) =>
        item.title &&
        item.link &&
        !QUOTE_PAGE_TITLE.test(item.title) &&
        matcher.test(item.title),
    )
    .slice(0, limit)
    .map((item) => ({
      title: item.title,
      description: "",
      link: item.link,
      publishedAt: item.providerPublishTime
        ? new Date(item.providerPublishTime).toUTCString()
        : null,
      source: item.publisher || "Yahoo Finance",
    }));
}
