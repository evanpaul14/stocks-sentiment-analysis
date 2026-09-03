import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getStockPageData } from "@/lib/stock/getStockPageData";
import { SymbolNotFoundError } from "@/lib/integrations/yahoo/search";
import { PriceChart } from "@/components/stock/PriceChart";
import { LivePrice } from "@/components/stock/LivePrice";
import { SentimentStream } from "@/components/stock/SentimentStream";
import { WatchlistToggleButton } from "@/components/watchlist/WatchlistToggleButton";
import { MovementInsight } from "@/components/stock/MovementInsight";
import { StockTwitsCard } from "@/components/stock/StockTwitsCard";
import { SentimentPriceOverlaySection } from "@/components/stock/SentimentPriceOverlaySection";

interface StockPageProps {
  params: Promise<{ symbol: string }>;
}

export async function generateMetadata({ params }: StockPageProps): Promise<Metadata> {
  const { symbol } = await params;
  const canonical = `/stock/${symbol.toUpperCase()}`;
  try {
    const { stockInfo } = await getStockPageData(symbol);
    const title = `${stockInfo.companyName} (${stockInfo.symbol}) Stock Price & Sentiment`;
    const description = `Live price, historical chart, and AI-powered news sentiment analysis for ${stockInfo.companyName} (${stockInfo.symbol}).`;
    return {
      title,
      description,
      alternates: { canonical },
      openGraph: { title, description, url: canonical, type: "website" },
      twitter: { card: "summary_large_image", title, description },
    };
  } catch {
    return { title: `${symbol.toUpperCase()} — Stock Sentiment`, alternates: { canonical } };
  }
}

export default async function StockPage({ params }: StockPageProps) {
  const { symbol } = await params;

  let data;
  try {
    data = await getStockPageData(symbol);
  } catch (error) {
    if (error instanceof SymbolNotFoundError) notFound();
    throw error;
  }

  const { stockInfo, historicalData, articles } = data;

  return (
    <main className="mx-auto max-w-3xl px-4 py-10">
      <header className="mb-6">
        <p className="text-sm text-muted-foreground">{stockInfo.symbol}</p>
        <h1 className="text-2xl font-semibold">{stockInfo.companyName}</h1>
        <div className="mt-2 flex items-center justify-between gap-3">
          <LivePrice
            symbol={stockInfo.symbol}
            initialPrice={stockInfo.currentPrice}
            initialChangePercent={stockInfo.regularMarketChangePercent}
          />
          <WatchlistToggleButton
            symbol={stockInfo.symbol}
            companyName={stockInfo.companyName}
            price={stockInfo.currentPrice}
            changePercent={stockInfo.regularMarketChangePercent}
          />
        </div>
      </header>

      <section className="mb-8 rounded-xl border border-border bg-card p-4">
        <PriceChart data={historicalData} />
      </section>

      <MovementInsight
        symbol={stockInfo.symbol}
        companyName={stockInfo.companyName}
        changePercent={stockInfo.regularMarketChangePercent}
      />

      <section className="mb-8 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3">
        <StatTile label="Market Cap" value={formatLargeNumber(stockInfo.marketCap)} />
        <StatTile label="P/E Ratio" value={stockInfo.peRatio?.toFixed(2) ?? "—"} />
        <StatTile label="Day Range" value={formatRange(stockInfo.dayLow, stockInfo.dayHigh)} />
        <StatTile
          label="52wk Range"
          value={formatRange(stockInfo.fiftyTwoWeekLow, stockInfo.fiftyTwoWeekHigh)}
        />
        <StatTile label="Volume" value={formatLargeNumber(stockInfo.volume)} />
        <StatTile label="Sector" value={stockInfo.sector ?? "—"} />
      </section>

      <section className="mb-8">
        <h2 className="mb-3 text-lg font-medium">News Sentiment</h2>
        <SentimentStream
          ticker={stockInfo.symbol}
          companyName={stockInfo.companyName}
          articles={articles}
        />
      </section>

      <StockTwitsCard symbol={stockInfo.symbol} />
      <SentimentPriceOverlaySection symbol={stockInfo.symbol} />
    </main>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-medium">{value}</p>
    </div>
  );
}

function formatRange(low: number | null, high: number | null): string {
  if (low == null || high == null) return "—";
  return `$${low.toFixed(2)} – $${high.toFixed(2)}`;
}

function formatLargeNumber(value: number | null): string {
  if (value == null) return "—";
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return value.toLocaleString();
}
