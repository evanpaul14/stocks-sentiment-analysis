export interface NewsArticle {
  title: string;
  description: string;
  link: string;
  publishedAt: string | null;
  source: string;
}

export function stripHtml(html: string): string {
  return html
    .replace(/<[^>]*>/g, " ")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
