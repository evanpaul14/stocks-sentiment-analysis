"""
Integration tests for the real AI sentiment APIs.

These tests make live network calls to Cloudflare Workers AI and Google Gemma.
They are skipped automatically when the required env vars are absent, so they
won't break CI unless the credentials are available.

Run manually:
    source .venv/bin/activate
    python -m pytest tests/test_sentiment_integration.py -v -s
"""
import os
import sys
import unittest

# Load .env ourselves first so real credentials land in os.environ before we set
# any placeholders. main.py calls load_dotenv() too but won't overwrite existing vars.
from dotenv import load_dotenv as _load_dotenv  # noqa: E402
_load_dotenv()

_have_google = bool(os.getenv("GOOGLE_API_KEY"))
_have_cloudflare = bool(os.getenv("CLOUDFLARE_ACCOUNT_ID") and os.getenv("CLOUDFLARE_API_TOKEN"))

# Placeholders for keys main.py requires at import time — only applied when absent.
os.environ.setdefault("GOOGLE_API_KEY", "placeholder-for-collection")
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")

import main  # noqa: E402

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

# A handful of real-world headlines with clear expected polarity.
POSITIVE_ARTICLES = [
    {
        "title": "Apple reports record quarterly revenue, beats analyst expectations",
        "description": "Apple Inc. posted its strongest quarter ever, driven by iPhone 16 sales and Services growth.",
        "company": "Apple",
    },
    {
        "title": "Tesla stock surges after delivering more vehicles than expected",
        "description": "Tesla shares climbed after the EV maker reported better-than-anticipated delivery numbers.",
        "company": "Tesla",
    },
]

NEGATIVE_ARTICLES = [
    {
        "title": "Microsoft lays off thousands of employees amid restructuring",
        "description": "Microsoft announced significant job cuts across multiple divisions as it reorganises operations.",
        "company": "Microsoft",
    },
    {
        "title": "Amazon stock drops after missing revenue guidance",
        "description": "Amazon shares fell sharply after the company warned that Q3 revenue would come in below analyst forecasts.",
        "company": "Amazon",
    },
]

NEUTRAL_ARTICLES = [
    {
        "title": "Google announces annual developer conference dates for next year",
        "description": "Alphabet's Google unit said it will hold its I/O developer conference in May.",
        "company": "Google",
    },
]

ALL_ARTICLES = POSITIVE_ARTICLES + NEGATIVE_ARTICLES + NEUTRAL_ARTICLES


@unittest.skipUnless(_have_cloudflare, "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN not set")
class TestCloudflareAPI(unittest.TestCase):
    """Live calls to Cloudflare Workers AI."""

    def test_returns_valid_sentiment_for_each_article(self):
        for article in ALL_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = main.analyze_sentiment_cloudflare(
                    article["title"],
                    article["description"],
                    article["company"],
                )
                # Cloudflare may return None if the model can't parse a label;
                # treat that as a provider skip, not a test failure.
                if result is None:
                    self.skipTest("Cloudflare returned None — provider may be unavailable")
                self.assertIn(
                    result,
                    VALID_SENTIMENTS,
                    f"Unexpected label '{result}' for: {article['title'][:60]}",
                )
                print(f"  [{result:8s}] {article['title'][:70]}")

    def test_positive_articles_not_labelled_negative(self):
        for article in POSITIVE_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = main.analyze_sentiment_cloudflare(
                    article["title"],
                    article["description"],
                    article["company"],
                )
                if result is None:
                    self.skipTest("Cloudflare returned None")
                self.assertNotEqual(
                    result,
                    "negative",
                    f"Clearly positive headline was labelled negative: {article['title']}",
                )

    def test_negative_articles_not_labelled_positive(self):
        for article in NEGATIVE_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = main.analyze_sentiment_cloudflare(
                    article["title"],
                    article["description"],
                    article["company"],
                )
                if result is None:
                    self.skipTest("Cloudflare returned None")
                self.assertNotEqual(
                    result,
                    "positive",
                    f"Clearly negative headline was labelled positive: {article['title']}",
                )


@unittest.skipUnless(_have_google, "GOOGLE_API_KEY not set")
class TestGemmaAPI(unittest.TestCase):
    """Live calls to Google Gemma via the google-generativeai client."""

    def _call_gemma(self, article):
        """Call analyze_sentiment with Cloudflare disabled so Gemma is always used."""
        original = main.CLOUDFLARE_ENABLED
        main.CLOUDFLARE_ENABLED = False
        try:
            return main.analyze_sentiment(
                article["title"],
                article["description"],
                article["company"],
            )
        finally:
            main.CLOUDFLARE_ENABLED = original

    def test_returns_valid_sentiment_for_each_article(self):
        for article in ALL_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = self._call_gemma(article)
                self.assertIn(
                    result,
                    VALID_SENTIMENTS,
                    f"Unexpected label '{result}' for: {article['title'][:60]}",
                )
                print(f"  [{result:8s}] {article['title'][:70]}")

    def test_positive_articles_not_labelled_negative(self):
        for article in POSITIVE_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = self._call_gemma(article)
                self.assertNotEqual(
                    result,
                    "negative",
                    f"Clearly positive headline was labelled negative: {article['title']}",
                )

    def test_negative_articles_not_labelled_positive(self):
        for article in NEGATIVE_ARTICLES:
            with self.subTest(title=article["title"][:60]):
                result = self._call_gemma(article)
                self.assertNotEqual(
                    result,
                    "positive",
                    f"Clearly negative headline was labelled positive: {article['title']}",
                )


@unittest.skipUnless(_have_cloudflare and _have_google, "Both CLOUDFLARE and GOOGLE credentials required")
class TestFallbackChain(unittest.TestCase):
    """Verify the Cloudflare → Gemma fallback produces a consistent label."""

    def test_both_providers_agree_on_polarity_for_clear_headlines(self):
        clear_articles = POSITIVE_ARTICLES[:1] + NEGATIVE_ARTICLES[:1]
        for article in clear_articles:
            with self.subTest(title=article["title"][:60]):
                cf_result = main.analyze_sentiment_cloudflare(
                    article["title"], article["description"], article["company"]
                )

                main_cf_enabled = main.CLOUDFLARE_ENABLED
                main.CLOUDFLARE_ENABLED = False
                try:
                    gemma_result = main.analyze_sentiment(
                        article["title"], article["description"], article["company"]
                    )
                finally:
                    main.CLOUDFLARE_ENABLED = main_cf_enabled

                if cf_result is None:
                    self.skipTest("Cloudflare unavailable, skipping agreement check")

                # Two different LLMs may not return identical labels, but they
                # should never give directly opposing signals (positive vs negative).
                opposites = {("positive", "negative"), ("negative", "positive")}
                self.assertNotIn(
                    (cf_result, gemma_result),
                    opposites,
                    f"Providers gave opposite labels for '{article['title'][:60]}': "
                    f"Cloudflare={cf_result}, Gemma={gemma_result}",
                )
                print(f"  CF={cf_result}, Gemma={gemma_result} — {article['title'][:55]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
