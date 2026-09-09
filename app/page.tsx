import type { Metadata } from "next";
import { SearchBar } from "@/components/search/SearchBar";
import { ParticleFlowField } from "@/components/home/ParticleFlowField";
import { PanelSwitcher } from "@/components/home/PanelSwitcher";
import { PopularSentimentTickers } from "@/components/home/PopularSentimentTickers";

export const metadata: Metadata = {
  alternates: { canonical: "/" },
};

export default function Home() {
  return (
    <main className="relative flex min-h-[calc(100vh-4rem)] flex-col items-center justify-center gap-8 overflow-hidden px-4 py-20">
      <div className="absolute inset-x-0 top-0">
        <PopularSentimentTickers />
      </div>

      <ParticleFlowField />

      <div className="relative z-10 text-center">
        <h1 className="font-serif text-4xl font-medium tracking-tight text-balance sm:text-5xl">
          Read the market&rsquo;s mood
        </h1>
        <p className="mt-2 font-serif text-lg text-primary italic">
          Signal, not noise.
        </p>
        <p className="mx-auto mt-3 max-w-md text-muted-foreground">
          Real-time prices and AI-powered news sentiment for any stock.
        </p>
      </div>

      <div className="relative z-10 flex w-full justify-center">
        <SearchBar />
      </div>

      <div className="relative z-10 flex w-full justify-center">
        <PanelSwitcher />
      </div>
    </main>
  );
}
