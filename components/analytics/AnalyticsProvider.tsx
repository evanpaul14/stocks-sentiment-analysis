import Script from "next/script";
import { getCspNonce } from "@/lib/seo/JsonLd";

/**
 * Umami by default (self-hosted-friendly, matches the VPS deployment story).
 * Swap the script tag here to change providers without touching call sites —
 * event tracking elsewhere uses the generic `data-umami-event` attribute
 * convention, which most privacy-friendly analytics tools also support.
 */
export async function AnalyticsProvider() {
  const scriptUrl = process.env.UMAMI_SCRIPT_URL;
  const websiteId = process.env.UMAMI_WEBSITE_ID;
  if (!scriptUrl || !websiteId) return null;

  const nonce = await getCspNonce();

  return (
    <Script
      src={scriptUrl}
      data-website-id={websiteId}
      strategy="afterInteractive"
      nonce={nonce}
    />
  );
}
