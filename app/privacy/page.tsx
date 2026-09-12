import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "How Stock Sentiment collects, uses, and protects your data.",
  alternates: { canonical: "/privacy" },
};

const LAST_UPDATED = "September 11, 2026";

export default function PrivacyPage() {
  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-2 text-2xl font-semibold">Privacy Policy</h1>
      <p className="mb-6 text-xs text-muted-foreground">Last updated: {LAST_UPDATED}</p>

      <div className="prose prose-invert max-w-none space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          Stock Sentiment (&ldquo;we,&rdquo; &ldquo;us,&rdquo; or &ldquo;our&rdquo;) is an
          independent, single-developer service (the &ldquo;Service&rdquo;). This Privacy Policy
          describes the information we collect in connection with the Service, the purposes for
          which it is used, and the choices available to users. The core functionality of the
          Service — retrieving prices, charts, and news sentiment — may be used without creating
          an account or providing any personal information.
        </p>

        <h2 className="text-base font-medium text-foreground">Information we collect</h2>
        <p>
          <strong className="text-foreground">Anonymous visitors.</strong> Users who do not create
          an account have their watchlist and recent search history stored exclusively in their
          browser&rsquo;s local storage. This information is not received or stored on our
          servers, except insofar as it is transmitted as part of the ordinary requests required
          to retrieve information about the tickers searched for (which necessarily include the
          user&rsquo;s IP address, as with any web request).
        </p>
        <p>
          <strong className="text-foreground">Account holders.</strong> Users who create an
          account have their authentication handled entirely by our identity provider, Supabase.
          Depending on the method of registration, this means we (via Supabase) hold the
          user&rsquo;s email address and a securely hashed password, or, where &ldquo;Sign in with
          Google&rdquo; is used, the basic profile information Google makes available for that
          purpose (name, email address, and profile photograph). We do not access or store
          passwords directly. Following sign-in, a user&rsquo;s watchlist and search history are
          stored in our database and associated with the account, enabling synchronization across
          devices. Upon first sign-in, any watchlist or search history previously stored in the
          user&rsquo;s browser is merged into the account on a one-time basis.
        </p>
        <p>
          <strong className="text-foreground">Email subscribers.</strong> Users who subscribe to
          the daily market summary email have their email address stored with our email delivery
          provider, Mailgun, solely for the purpose of delivering that digest. Subscribers may
          unsubscribe at any time using the link provided in each email, or through their account
          settings if signed in.
        </p>
        <p>
          <strong className="text-foreground">Contact form.</strong> Users who submit the contact
          form provide their name, email address, and message, which are transmitted to us via
          Mailgun so that we may respond. Submissions are protected by Cloudflare Turnstile, a
          bot-verification service that receives the submitting user&rsquo;s IP address and
          browser signals in order to verify human interaction; further information is available
          in{" "}
          <a
            href="https://www.cloudflare.com/privacypolicy/"
            target="_blank"
            rel="noreferrer"
            className="text-foreground underline"
          >
            Cloudflare&rsquo;s privacy policy
          </a>
          .
        </p>
        <p>
          <strong className="text-foreground">Server logs and rate limiting.</strong> As is
          standard for web services, our server briefly processes IP addresses and request
          metadata for the purpose of preventing abuse (for example, to limit the frequency with
          which a given address may submit the contact form or query the sentiment API). This
          information is retained only in server memory for rate-limiting purposes, is not
          persisted to a database, and is not used to construct profiles of individual visitors.
        </p>

        <h2 className="text-base font-medium text-foreground">How we use your information</h2>
        <p>
          The information described above is used solely to: operate the features a user elects
          to use (accounts, watchlists, search history, and email digests); respond to
          correspondence submitted through the contact form; secure the Service against abuse;
          and understand aggregate usage of the Service (see &ldquo;Analytics&rdquo; below). We do
          not sell personal information, do not use it for advertising, and do not disclose it to
          third parties other than the service providers identified in this Privacy Policy, which
          process such information solely on our behalf and for the purpose of providing the
          Service.
        </p>

        <h2 className="text-base font-medium text-foreground">Legal basis for processing</h2>
        <p>
          For users located in the European Economic Area, the United Kingdom, or another
          jurisdiction requiring a stated legal basis for processing, we rely on the following
          bases: performance of a contract with the user (creating and operating an account,
          synchronizing watchlist and search history); the user&rsquo;s consent (subscribing to
          the market summary email, submitting the contact form); and our legitimate interest in
          operating, securing, and improving the Service (rate-limiting abuse, aggregate
          analytics). Consent may be withdrawn at any time, as described under &ldquo;Your choices
          and rights&rdquo; below, without affecting the lawfulness of processing conducted prior
          to such withdrawal.
        </p>

        <h2 className="text-base font-medium text-foreground">Cookies and similar technology</h2>
        <p>
          Where an account is created, Supabase sets essential cookies to maintain the
          user&rsquo;s signed-in session. These cookies are required for the account feature to
          function and are not used for tracking purposes. We do not employ advertising or
          cross-site tracking cookies.
        </p>

        <h2 className="text-base font-medium text-foreground">Analytics</h2>
        <p>
          We employ Umami, a cookie-free analytics tool, to understand aggregate usage of the
          Service (for example, which pages are most frequently visited). Analytics are provided
          through Umami Cloud, a hosted service operated by Umami Software, Inc., which acts as a
          processor on our behalf; aggregate page-view data is accordingly transmitted to and
          stored on that provider&rsquo;s infrastructure rather than on our own servers. We have
          selected that provider&rsquo;s European Union region, and such data is therefore stored
          on infrastructure located within the EU. Umami does not use cookies or persistent
          identifiers, does not retain visitor IP addresses, and does not track users across
          other websites; accordingly, no individual visitor profile is created.
        </p>

        <h2 className="text-base font-medium text-foreground">Third-party data sources</h2>
        <p>
          Stock prices, news headlines, and social sentiment displayed on the Service are sourced
          from third-party providers, including Yahoo Finance, Google News, StockTwits, Reddit
          (via ApeWisdom), and Alpaca. These providers are queried for the purpose of displaying
          information to users. We are not responsible for the data practices of these providers
          or for the accuracy of the data they supply, and use of the Service does not create a
          direct relationship between the user and any such provider.
        </p>

        <h2 className="text-base font-medium text-foreground">Data retention</h2>
        <p>
          Account data (watchlist, search history, and email address) is retained for as long as
          the account remains active. Contact form submissions are retained only for as long as
          necessary to respond to and resolve the relevant inquiry. Upon deletion of an account,
          the associated account record and data (watchlist and search history) are permanently
          and irrevocably deleted.
        </p>

        <h2 className="text-base font-medium text-foreground">Your choices and rights</h2>
        <p>
          Account holders may view and delete their watchlist and search history, and may
          permanently delete their account and all associated data, through their{" "}
          <Link href="/account" className="text-foreground underline">
            account settings
          </Link>
          . Subscribers may unsubscribe from the market summary email at any time using the link
          provided in each email. Depending on the user&rsquo;s jurisdiction, additional rights
          with respect to personal information may apply, including the right to access, correct,
          delete, or obtain a copy of such information, and the right to object to or restrict
          certain processing (for example, under the EU/UK General Data Protection Regulation or
          applicable U.S. state privacy laws). Any such right may be exercised by contacting us as
          described below; we will respond within a reasonable period, consistent with applicable
          law.
        </p>

        <h2 className="text-base font-medium text-foreground">Data security</h2>
        <p>
          We employ industry-standard measures to protect user information, including encrypted
          connections (HTTPS) and delegation of password storage and authentication to Supabase
          rather than handling credentials directly. No method of transmission or storage is
          entirely secure, and no guarantee of absolute security can be made.
        </p>

        <h2 className="text-base font-medium text-foreground">Children&rsquo;s privacy</h2>
        <p>
          The Service is not directed to children under the age of thirteen, and we do not
          knowingly collect personal information from children under thirteen. Any person who
          believes that a child has provided us with personal information is asked to contact us,
          and such information will be deleted.
        </p>

        <h2 className="text-base font-medium text-foreground">International visitors</h2>
        <p>
          We operate from the United States, and all information collected is processed and
          stored on servers located within the United States. Users accessing the Service from
          outside the United States acknowledge that their information will be transferred to and
          processed in the United States, which may afford different data protection standards
          than those of their country of residence.
        </p>

        <h2 className="text-base font-medium text-foreground">Governing law</h2>
        <p>
          This Privacy Policy is governed by the laws of the United States, without regard to
          conflict-of-law principles, except to the extent that applicable law (including the
          GDPR or a U.S. state privacy law) confers rights upon a user that may not be waived by
          this provision, in which case such rights remain in full force and effect.
        </p>

        <h2 className="text-base font-medium text-foreground">Changes to this policy</h2>
        <p>
          This Privacy Policy may be updated from time to time. Material changes will be reflected
          by an update to the &ldquo;Last updated&rdquo; date above. Continued use of the Service
          following any such change constitutes acceptance of the updated Privacy Policy.
        </p>

        <h2 className="text-base font-medium text-foreground">Contact</h2>
        <p>
          Questions regarding this Privacy Policy, or requests to access, correct, or delete
          personal information, may be directed to us via the{" "}
          <Link href="/contact" className="text-foreground underline">
            contact form
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
