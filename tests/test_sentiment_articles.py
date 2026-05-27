"""
Test that sentiment analysis runs successfully for each of the 10 articles
returned for a stock ticker, exercising the /sentiment endpoint.

AI providers (Cloudflare and Gemma) are mocked so no real API calls are made.
"""
import importlib
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Patch env vars before main.py is imported (it raises on missing keys).
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-api-key")

# Remove cached module so env-var patches take effect on fresh import.
for mod in list(sys.modules.keys()):
    if mod == "main" or mod.startswith("main."):
        del sys.modules[mod]

# Stub out google.generativeai (genai) before main imports it.
genai_stub = MagicMock()
genai_stub.Client.return_value = MagicMock()
sys.modules.setdefault("google.generativeai", genai_stub)

import main  # noqa: E402  (must come after env + stub setup)

TICKER = "AAPL"
COMPANY = "Apple"

FAKE_ARTICLES = [
    {
        "title": f"Apple news headline {i}",
        "description": f"Description of Apple article {i}.",
        "link": f"https://example.com/article-{i}",
        "publishedAt": "2026-05-26T10:00:00Z",
        "source": "Reuters",
    }
    for i in range(1, 11)
]

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


class TestSentimentForTenArticles(unittest.TestCase):

    def setUp(self):
        main.app.config["TESTING"] = True
        self.client = main.app.test_client()
        # Reset in-memory rate-limit counters so tests don't interfere with each other.
        main.limiter.reset()

    def _post_sentiment(self, article):
        return self.client.post(
            "/sentiment",
            data=json.dumps(
                {
                    "title": article["title"],
                    "description": article["description"],
                    "symbol": TICKER,
                    "company_name": COMPANY,
                    "article_id": article["link"],
                }
            ),
            content_type="application/json",
        )

    @patch("main.analyze_sentiment_cloudflare")
    def test_all_ten_articles_return_valid_sentiment(self, mock_cf):
        """Each article gets a 200 response with a recognised sentiment label."""
        mock_cf.return_value = "positive"

        for i, article in enumerate(FAKE_ARTICLES):
            with self.subTest(article_index=i):
                resp = self._post_sentiment(article)
                self.assertEqual(resp.status_code, 200, f"article {i}: unexpected status")
                body = resp.get_json()
                self.assertIn("sentiment", body, f"article {i}: missing 'sentiment' key")
                self.assertIn(
                    body["sentiment"],
                    VALID_SENTIMENTS,
                    f"article {i}: '{body['sentiment']}' is not a valid sentiment",
                )
                self.assertEqual(
                    body.get("article_id"),
                    article["link"],
                    f"article {i}: article_id not echoed back",
                )

    @patch("main.analyze_sentiment_cloudflare")
    def test_mixed_sentiments_across_articles(self, mock_cf):
        """Cloudflare can return different sentiments; all should be accepted."""
        labels = ["positive", "negative", "neutral"] * 4  # 12 items; slice to 10
        mock_cf.side_effect = labels[:10]

        for i, article in enumerate(FAKE_ARTICLES):
            with self.subTest(article_index=i):
                resp = self._post_sentiment(article)
                self.assertEqual(resp.status_code, 200)
                self.assertIn(resp.get_json()["sentiment"], VALID_SENTIMENTS)

    @patch("main.analyze_sentiment_cloudflare")
    @patch("main.wait_for_ai_rate_slot")
    def test_falls_back_to_gemma_when_cloudflare_fails(self, mock_rate, mock_cf):
        """If Cloudflare returns None the endpoint should fall back to Gemma."""
        mock_cf.return_value = None  # force fallback path

        gemma_response = MagicMock()
        gemma_response.text = "positive"
        main.client.models.generate_content = MagicMock(return_value=gemma_response)

        for i, article in enumerate(FAKE_ARTICLES):
            with self.subTest(article_index=i):
                resp = self._post_sentiment(article)
                self.assertEqual(resp.status_code, 200)
                self.assertIn(resp.get_json()["sentiment"], VALID_SENTIMENTS)

    @patch("main.analyze_sentiment_cloudflare")
    def test_missing_title_returns_400(self, mock_cf):
        """A request without a title should fail validation before hitting AI."""
        resp = self.client.post(
            "/sentiment",
            data=json.dumps({"description": "some text", "symbol": TICKER}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        mock_cf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
