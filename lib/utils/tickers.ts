export interface SeoSentimentCompany {
  ticker: string;
  companyName: string;
  groups: string[];
  /** Ticker to show in UI when it differs from the Yahoo Finance lookup symbol (e.g. index tickers). */
  displayTicker?: string;
}

// The old "mag7"/"faang" groups overlapped on 4 of 7 companies (AAPL, AMZN,
// GOOGL, META were in both), so relatedCompanies() kept surfacing the same
// handful of tickers for each other while NFLX/TSLA/NVDA rarely got linked
// from anywhere. Replaced with two disjoint, still-coherent clusters so
// every company has a clear, non-overlapping set of peers.
export const SEO_SENTIMENT_COMPANIES: SeoSentimentCompany[] = [
  { ticker: "AAPL", companyName: "Apple", groups: ["cluster-hardware-ai"] },
  { ticker: "MSFT", companyName: "Microsoft", groups: ["cluster-hardware-ai"] },
  { ticker: "NVDA", companyName: "NVIDIA", groups: ["cluster-hardware-ai"] },
  { ticker: "TSLA", companyName: "Tesla", groups: ["cluster-hardware-ai"] },
  { ticker: "AMZN", companyName: "Amazon", groups: ["cluster-consumer-media"] },
  { ticker: "GOOGL", companyName: "Alphabet", groups: ["cluster-consumer-media"] },
  { ticker: "META", companyName: "Meta", groups: ["cluster-consumer-media"] },
  { ticker: "NFLX", companyName: "Netflix", groups: ["cluster-consumer-media"] },
  { ticker: "^DJI", companyName: "Dow Jones Industrial Average", groups: ["index"] },
  { ticker: "^IXIC", companyName: "Nasdaq Composite", groups: ["index"] },
  { ticker: "^GSPC", companyName: "S&P 500", groups: ["index"], displayTicker: "SPX" },
];

/** UI-facing ticker: uses `displayTicker` when set, otherwise strips a leading "^" (index symbols). */
export function displayTicker(company: SeoSentimentCompany): string {
  return company.displayTicker ?? company.ticker.replace(/^\^/, "");
}

/** Whether this SEO company is a market index (^DJI, ^IXIC, ^GSPC) rather than a single stock. */
export function isIndexCompany(company: SeoSentimentCompany): boolean {
  return company.groups.includes("index");
}

function slugify(companyName: string): string {
  return companyName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

export function companySlug(companyName: string): string {
  return `sentiment-of-${slugify(companyName)}-stock`;
}

export function findSeoCompanyBySlug(slug: string): SeoSentimentCompany | undefined {
  return SEO_SENTIMENT_COMPANIES.find((c) => companySlug(c.companyName) === slug);
}

export function relatedCompanies(company: SeoSentimentCompany, limit = 4): SeoSentimentCompany[] {
  return SEO_SENTIMENT_COMPANIES.filter(
    (c) => c.ticker !== company.ticker && c.groups.some((g) => company.groups.includes(g))
  ).slice(0, limit);
}
