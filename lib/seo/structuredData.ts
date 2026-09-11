const baseUrl = () => process.env.SITE_BASE_URL ?? "http://localhost:3000";

export function organizationJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "Stock Sentiment",
    alternateName: "StockSentimentApp",
    url: baseUrl(),
  };
}

export function websiteJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "Stock Sentiment",
    alternateName: "StockSentimentApp",
    url: baseUrl(),
    potentialAction: {
      "@type": "SearchAction",
      target: `${baseUrl()}/stock/{search_term_string}`,
      "query-input": "required name=search_term_string",
    },
  };
}

export function breadcrumbListJsonLd(items: Array<{ name: string; url: string }>) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.name,
      item: item.url,
    })),
  };
}

export function faqPageJsonLd(questions: Array<{ question: string; answer: string }>) {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: questions.map((q) => ({
      "@type": "Question",
      name: q.question,
      acceptedAnswer: { "@type": "Answer", text: q.answer },
    })),
  };
}

export function articleJsonLd(options: {
  headline: string;
  description: string;
  datePublished: string;
  dateModified?: string;
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
    dateModified: options.dateModified ?? options.datePublished,
    url: options.url,
    author: { "@type": "Organization", name: options.author ?? "Stock Sentiment Team" },
    image: options.image ?? `${baseUrl()}/logo-icon-512.png`,
  };
}
