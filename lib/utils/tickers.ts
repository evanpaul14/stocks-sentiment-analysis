export const MAG7_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"];
export const FAANG_TICKERS = ["META", "AMZN", "AAPL", "NFLX", "GOOGL"];

export interface SeoSentimentCompany {
  ticker: string;
  companyName: string;
  groups: string[];
  /** Ticker to show in UI when it differs from the Yahoo Finance lookup symbol (e.g. index tickers). */
  displayTicker?: string;
}

/** Static config for programmatic SEO sentiment pages (/blog/sentiment-of-<company>-stock). */
export const SEO_SENTIMENT_COMPANIES: SeoSentimentCompany[] = [
  { ticker: "AAPL", companyName: "Apple", groups: ["mag7", "faang"] },
  { ticker: "MSFT", companyName: "Microsoft", groups: ["mag7"] },
  { ticker: "AMZN", companyName: "Amazon", groups: ["mag7", "faang"] },
  { ticker: "GOOGL", companyName: "Alphabet", groups: ["mag7", "faang"] },
  { ticker: "META", companyName: "Meta", groups: ["mag7", "faang"] },
  { ticker: "NVDA", companyName: "NVIDIA", groups: ["mag7"] },
  { ticker: "TSLA", companyName: "Tesla", groups: ["mag7"] },
  { ticker: "NFLX", companyName: "Netflix", groups: ["faang"] },
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
