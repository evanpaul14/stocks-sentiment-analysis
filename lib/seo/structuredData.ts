const baseUrl = () => process.env.SITE_BASE_URL ?? "http://localhost:3000";

export function organizationJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "Stock Sentiment",
    url: baseUrl(),
  };
}

export function websiteJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "Stock Sentiment",
    url: baseUrl(),
    potentialAction: {
      "@type": "SearchAction",
      target: `${baseUrl()}/stock/{search_term_string}`,
      "query-input": "required name=search_term_string",
    },
  };
}

export function articleJsonLd(options: {
  headline: string;
  description: string;
  datePublished: string;
  url: string;
  author?: string;
  image?: string;
}) {
  return {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: options.headline,
    description: options.description,
    datePublished: options.datePublished,
    url: options.url,
    author: { "@type": "Organization", name: options.author ?? "Stock Sentiment Team" },
    ...(options.image ? { image: options.image } : {}),
  };
}
