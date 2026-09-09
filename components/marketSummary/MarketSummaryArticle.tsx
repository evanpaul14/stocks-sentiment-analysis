import Image from "next/image";
import { JsonLd } from "@/lib/seo/JsonLd";
import { articleJsonLd } from "@/lib/seo/structuredData";

interface IndexSnapshot {
  symbol: string;
  name: string;
  price: number | null;
  changePercent: number | null;
}

interface MarketSummaryArticleProps {
  title: string;
  body: string;
  indexSnapshotJson: string;
  createdAt: string;
  slug: string;
  imageUrl?: string | null;
  imagePhotographerName?: string | null;
  imagePhotographerProfileUrl?: string | null;
}

export function MarketSummaryArticle({
  title,
  body,
  indexSnapshotJson,
  createdAt,
  slug,
  imageUrl,
  imagePhotographerName,
  imagePhotographerProfileUrl,
}: MarketSummaryArticleProps) {
  let indexes: IndexSnapshot[] = [];
  try {
    indexes = JSON.parse(indexSnapshotJson);
  } catch {
    indexes = [];
  }

  return (
    <article>
      <JsonLd
        data={articleJsonLd({
          headline: title,
          description: body.slice(0, 200),
          datePublished: createdAt,
          url: `${process.env.SITE_BASE_URL ?? ""}/market-summary/${slug}`,
          image: imageUrl ?? undefined,
        })}
      />
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="mt-1 text-xs text-muted-foreground">
        {new Date(createdAt).toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })}
      </p>

      {imageUrl && (
        <figure className="mt-4 w-full">
          <Image
            src={imageUrl}
            alt={title}
            width={1280}
            height={720}
            className="aspect-video w-full rounded-xl border border-border object-cover"
          />
          {imagePhotographerName && (
            <figcaption className="mt-1 text-right text-xs text-muted-foreground">
              Photo by{" "}
              {imagePhotographerProfileUrl ? (
                <a
                  href={imagePhotographerProfileUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline"
                >
                  {imagePhotographerName}
                </a>
              ) : (
                imagePhotographerName
              )}{" "}
              on Unsplash
            </figcaption>
          )}
        </figure>
      )}

      {indexes.length > 0 && (
        <div className="mt-4 grid grid-cols-3 gap-3">
          {indexes.map((index) => {
            const isUp = (index.changePercent ?? 0) >= 0;
            return (
              <div key={index.symbol} className="rounded-lg border border-border p-3">
                <p className="text-xs text-muted-foreground">{index.name}</p>
                <p className="font-medium tabular-nums">
                  {index.price != null ? index.price.toFixed(2) : "—"}
                </p>
                {index.changePercent != null && (
                  <p
                    className={`text-xs tabular-nums ${
                      isUp ? "text-[var(--color-chart-1)]" : "text-destructive"
                    }`}
                  >
                    {isUp ? "+" : ""}
                    {index.changePercent.toFixed(2)}%
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="prose prose-invert mt-6 max-w-none text-sm leading-relaxed">
        {body
          .split("\n")
          .filter(Boolean)
          .map((paragraph, i) => (
            <p key={i} className="mb-3">
              {paragraph}
            </p>
          ))}
      </div>
    </article>
  );
}
