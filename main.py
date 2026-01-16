from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, session
from flask_sqlalchemy import SQLAlchemy
from google import genai
from pygooglenews import GoogleNews
from yahooquery import search
import yfinance as yf
import re
import hashlib
import inspect as pyinspect
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
import os
import logging
import hmac
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
from collections import deque
import threading
from functools import lru_cache
import cloudscraper
from openai import OpenAI
import finnhub
import json
import atexit
import uuid
from zoneinfo import ZoneInfo
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, text, inspect
from sqlalchemy.exc import IntegrityError, OperationalError
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from pathlib import Path
import xml.etree.ElementTree as ET

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
SITEMAP_FILE_PATH = PROJECT_ROOT / "sitemap.xml"
MARKET_SUMMARY_SITEMAP_BASE_URL = (
    (os.getenv("MARKET_SUMMARY_SITEMAP_BASE_URL") or "https://stocksentimentapp.com/market-summary").rstrip("/")
    or "https://stocksentimentapp.com/market-summary"
)
SITEMAP_XML_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"
ET.register_namespace("", SITEMAP_XML_NAMESPACE)


log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level_name, logging.INFO))
summary_logger = logging.getLogger("stocktwits.summary")
market_summary_logger = logging.getLogger("market.summary")
unsplash_logger = logging.getLogger("unsplash")
app_logger = logging.getLogger("stocks.app")


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ip_log.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_BINDS'] = {
    'blog': 'sqlite:///blog.db'
}

_secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY")
if not _secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY (or SECRET_KEY) must be set so blog editor sessions remain valid across workers."
    )

app.secret_key = _secret_key
db = SQLAlchemy(app)

try:
    _CREATE_ALL_SIGNATURE = pyinspect.signature(SQLAlchemy.create_all)
except (ValueError, TypeError):
    _CREATE_ALL_SIGNATURE = None

_CREATE_ALL_SUPPORTS_BIND = bool(
    _CREATE_ALL_SIGNATURE and "bind" in _CREATE_ALL_SIGNATURE.parameters
)
_CREATE_ALL_SUPPORTS_BIND_KEY = bool(
    _CREATE_ALL_SIGNATURE and "bind_key" in _CREATE_ALL_SIGNATURE.parameters
)


HOME_META_DESCRIPTION = (
    "Get the ultimate market sentiment and stock sentiment analysis using AI to get live data and news analysis that help make stock predictions and stock forecasts."
)
TRENDING_META_DESCRIPTION = (
    "See trending stocks on Stocktwits, Reddit, and stock volume to see today's most active stocks."
)
MARKET_META_DESCRIPTION = (
    "Check the daily stock market summary with index moves, sector highlights, and curated headlines in one view."
)
STOCK_MARKET_TODAY_META_DESCRIPTION = (
    "Catch up on today's market movers, dow jones today, nasdaq today and s&p 500 performance with our stock market summary."
)

BLOG_ADMIN_USERNAME = os.getenv("BLOG_ADMIN_USERNAME")
BLOG_ADMIN_PASSWORD = os.getenv("BLOG_ADMIN_PASSWORD")
DEFAULT_ARTICLE_AUTHOR = os.getenv("BLOG_DEFAULT_AUTHOR", "stocksentimentapp.com Team")
BLOG_PAGE_META_DESCRIPTION = (
    "Draft and publish original market commentary with the built-in word processor."
)
BLOG_LIST_META_DESCRIPTION = (
    "Browse fresh market commentary, trade ideas, and sentiment takeaways from stocksentimentapp.com writers."
)


@app.context_processor
def inject_unsplash_globals():
    return {"unsplash_referral_url": _build_unsplash_referral_url()}


def _utcnow_naive():
    """Return a timezone-naive UTC datetime without using deprecated utcnow."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


EASTERN_TZ = ZoneInfo("America/New_York")


def _localize_datetime(value, target_tz):
    if not value or not target_tz:
        return None
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(target_tz)
    except Exception:
        return None


def format_blog_timestamp(value, include_time=True):
    localized = _localize_datetime(value, EASTERN_TZ)
    if not localized:
        return None
    if include_time:
        return localized.strftime('%b %d, %Y • %I:%M %p ET')
    return localized.strftime('%b %d, %Y')


app.jinja_env.filters['blog_timestamp'] = format_blog_timestamp


class ArticleImage(db.Model):
    __tablename__ = "article_images"

    id = db.Column(db.Integer, primary_key=True)
    article_key = db.Column(db.String(64), nullable=False, unique=True, index=True)
    article_url = db.Column(db.Text, nullable=True)
    query = db.Column(db.String(255), nullable=True)
    image_url = db.Column(db.String(512), nullable=False)
    thumbnail_url = db.Column(db.String(512), nullable=True)
    description = db.Column(db.String(512), nullable=True)
    photographer_name = db.Column(db.String(255), nullable=True)
    photographer_username = db.Column(db.String(255), nullable=True)
    photographer_profile_url = db.Column(db.String(512), nullable=True)
    unsplash_photo_id = db.Column(db.String(64), nullable=True, index=True)
    unsplash_photo_link = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive)
    updated_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive, onupdate=_utcnow_naive)

    def as_payload(self):
        return {
            "url": self.image_url,
            "thumbnail": self.thumbnail_url,
            "description": self.description,
            "photographer": self.photographer_name,
            "photographer_username": self.photographer_username,
            "photographer_profile": self.photographer_profile_url,
            "unsplash_link": self.unsplash_photo_link,
        }


class MarketSummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    body = db.Column(db.Text, nullable=False)
    summary_date = db.Column(db.Date, nullable=False, index=True)
    published_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive)
    updated_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive, onupdate=_utcnow_naive)
    index_snapshot = db.Column(db.Text)
    headline_sources = db.Column(db.Text)
    __table_args__ = (
        db.UniqueConstraint('summary_date', name='uq_market_summary_summary_date'),
    )


class MarketSummarySendLog(db.Model):
    __tablename__ = "market_summary_send_log"

    id = db.Column(db.Integer, primary_key=True)
    summary_id = db.Column(db.Integer, db.ForeignKey("market_summary.id"), nullable=False, unique=True)
    sent_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive)
    status = db.Column(db.String(32), nullable=False, default="sent")
    error_message = db.Column(db.String(255), nullable=True)


class BlogArticle(db.Model):
    __bind_key__ = "blog"
    __tablename__ = "blog_articles"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    slug = db.Column(db.String(255), nullable=False, unique=True, index=True)
    author = db.Column(db.String(255), nullable=False, default=DEFAULT_ARTICLE_AUTHOR)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive)
    updated_at = db.Column(db.DateTime, nullable=False, default=_utcnow_naive, onupdate=_utcnow_naive)
    is_published = db.Column(db.Boolean, nullable=False, default=False, server_default="0")
    published_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        permalink = None
        try:
            permalink = url_for('blog_article_page', slug=self.slug)
        except RuntimeError:
            permalink = f"/blog/{self.slug}"
        return {
            "id": self.id,
            "title": self.title,
            "slug": self.slug,
            "author": self.author,
            "content": self.content,
            "image_url": self.image_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_published": bool(self.is_published),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "permalink": permalink
        }


def _blog_credentials_configured():
    return bool(BLOG_ADMIN_USERNAME and BLOG_ADMIN_PASSWORD)


def _is_blog_admin():
    return bool(session.get('blog_admin_authenticated'))


def _verify_blog_credentials(username, password):
    if not _blog_credentials_configured():
        return False
    safe_username = username or ''
    safe_password = password or ''
    return (
        hmac.compare_digest(safe_username, BLOG_ADMIN_USERNAME)
        and hmac.compare_digest(safe_password, BLOG_ADMIN_PASSWORD)
    )


def _slugify_article_title(title):
    cleaned = (title or '').lower()
    cleaned = re.sub(r'[^a-z0-9]+', '-', cleaned)
    cleaned = cleaned.strip('-')
    if not cleaned:
        return f"article-{uuid.uuid4().hex[:8]}"
    return cleaned[:80]


def _generate_unique_blog_slug(title):
    base = _slugify_article_title(title)
    slug = base
    suffix = 2
    while BlogArticle.query.filter_by(slug=slug).first() is not None:
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def _create_blog_article(title, content, image_url=None, author=None):
    if not title or not content:
        raise ValueError("Title and content are required")
    normalized_author = (author or DEFAULT_ARTICLE_AUTHOR).strip() or DEFAULT_ARTICLE_AUTHOR
    slug = _generate_unique_blog_slug(title)
    article = BlogArticle(
        title=title.strip(),
        slug=slug,
        author=normalized_author,
        content=content,
        image_url=(image_url or '').strip() or None
    )
    db.session.add(article)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        article.slug = _generate_unique_blog_slug(f"{title}-{uuid.uuid4().hex[:4]}")
        db.session.add(article)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    return article


def _fetch_recent_blog_articles(limit=None):
    use_limit = limit or BLOG_ARTICLE_FETCH_LIMIT
    query = BlogArticle.query.order_by(BlogArticle.created_at.desc())
    if use_limit:
        query = query.limit(use_limit)
    return query.all()


def _delete_blog_article(identifier):
    article = _resolve_blog_article(identifier)
    if not article:
        return False
    try:
        db.session.delete(article)
        db.session.commit()
        return True
    except Exception as exc:
        db.session.rollback()
        app_logger.error("Unable to delete blog article %s: %s", identifier, exc)
        raise


def ensure_blog_article_columns():
    try:
        engine = db.get_engine(app, bind='blog')
    except Exception as exc:
        app_logger.warning("Unable to load blog engine for schema sync: %s", exc)
        return
    inspector = inspect(engine)
    existing_columns = {col['name'] for col in inspector.get_columns('blog_articles')}
    alter_statements = []
    if 'is_published' not in existing_columns:
        alter_statements.append(
            "ALTER TABLE blog_articles ADD COLUMN is_published INTEGER NOT NULL DEFAULT 0"
        )
    if 'published_at' not in existing_columns:
        alter_statements.append(
            "ALTER TABLE blog_articles ADD COLUMN published_at DATETIME"
        )
    if not alter_statements:
        return
    with engine.begin() as connection:
        for statement in alter_statements:
            connection.execute(text(statement))


def _create_tables_for_bind(bind_key):
    if not bind_key:
        raise ValueError("bind_key is required for bound table creation")
    try:
        engine = db.get_engine(app, bind=bind_key)
    except Exception as exc:
        app_logger.error("Unable to load engine for %s bind: %s", bind_key, exc)
        raise
    tables = [
        table
        for table in db.Model.metadata.sorted_tables
        if table.info.get("bind_key") == bind_key
    ]
    if not tables:
        app_logger.info("No tables registered for %s bind; skipping create_all", bind_key)
        return
    db.Model.metadata.create_all(bind=engine, tables=tables)


def _safe_create_all(bind=None):
    """Create tables for a bind while tolerating existing schemas."""
    bind_label = bind or "default"

    def _handle_operational_error(exc):
        message = str(exc).lower()
        if "already exists" in message:
            app_logger.info(
                "Skipping create_all for %s bind because tables already exist",
                bind_label
            )
            return
        raise

    def _invoke_create_all():
        if bind is None:
            db.create_all()
        elif _CREATE_ALL_SUPPORTS_BIND:
            db.create_all(bind=bind)
        elif _CREATE_ALL_SUPPORTS_BIND_KEY:
            db.create_all(bind_key=bind)
        else:
            _create_tables_for_bind(bind)

    try:
        _invoke_create_all()
    except OperationalError as exc:
        _handle_operational_error(exc)
    except TypeError as exc:
        if not bind:
            raise
        if "bind" not in str(exc).lower():
            raise
        try:
            _create_tables_for_bind(bind)
        except OperationalError as op_exc:
            _handle_operational_error(op_exc)


def _fetch_published_blog_articles(limit=None):
    query = (
        BlogArticle.query
        .filter_by(is_published=True)
        .order_by(BlogArticle.published_at.desc(), BlogArticle.created_at.desc())
    )
    if limit:
        query = query.limit(limit)
    return query.all()


def _fetch_blog_article_by_slug(slug):
    if not slug:
        return None
    return BlogArticle.query.filter_by(slug=slug, is_published=True).first()


def _resolve_blog_article(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, int):
        return db.session.get(BlogArticle, identifier)
    try:
        identifier_str = str(identifier).strip()
    except Exception:
        return None
    if not identifier_str:
        return None

    article = None
    if identifier_str.isdigit():
        try:
            article = db.session.get(BlogArticle, int(identifier_str))
        except Exception:
            article = None
        if article:
            return article

    return BlogArticle.query.filter_by(slug=identifier_str).first()


def _publish_blog_article(identifier):
    article = _resolve_blog_article(identifier)
    if not article:
        return None
    if article.is_published and article.published_at:
        return article
    article.is_published = True
    article.published_at = _utcnow_naive()
    try:
        db.session.add(article)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app_logger.error("Unable to publish blog article %s: %s", identifier, exc)
        raise
    return article


def _unpublish_blog_article(identifier):
    article = _resolve_blog_article(identifier)
    if not article:
        return None
    if not article.is_published:
        return article
    article.is_published = False
    article.published_at = None
    try:
        db.session.add(article)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app_logger.error("Unable to unpublish blog article %s: %s", identifier, exc)
        raise
    return article


def dedupe_market_summaries(summary_date=None):
    """Ensure only one MarketSummary row exists per date."""
    query = db.session.query(
        MarketSummary.summary_date,
        func.count(MarketSummary.id).label("row_count")
    )
    if summary_date is not None:
        query = query.filter(MarketSummary.summary_date == summary_date)

    duplicates = (
        query.group_by(MarketSummary.summary_date)
        .having(func.count(MarketSummary.id) > 1)
        .all()
    )

    removed = 0
    for duplicate_date, _ in duplicates:
        rows = (
            MarketSummary.query
            .filter_by(summary_date=duplicate_date)
            .order_by(MarketSummary.published_at.desc(), MarketSummary.id.desc())
            .all()
        )
        keeper = rows[0]
        for extra in rows[1:]:
            db.session.delete(extra)
            removed += 1
        market_summary_logger.warning(
            "Detected %s duplicate market summaries for %s; trimmed extras.",
            len(rows) - 1,
            duplicate_date
        )

    if removed:
        db.session.commit()
    return removed


def ensure_market_summary_unique_index():
    """Create a unique index on summary_date if the database allows it."""
    try:
        db.session.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_market_summary_summary_date "
                "ON market_summary (summary_date)"
            )
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        market_summary_logger.warning(
            "Unable to enforce unique index for market summaries: %s",
            exc
        )

with app.app_context():
    _safe_create_all()
    _safe_create_all(bind='blog')
    try:
        dedupe_market_summaries()
        ensure_market_summary_unique_index()
    except Exception as exc:
        market_summary_logger.warning(
            "Market summary uniqueness bootstrap failed: %s",
            exc
        )
    try:
        ensure_blog_article_columns()
    except Exception as exc:
        app_logger.warning(
            "Blog schema bootstrap failed: %s",
            exc
        )



# set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"] 
)

# Initialize Google Gemini client
apikey = os.getenv('GOOGLE_API_KEY')
if not apikey:
    raise ValueError("Secret key not set in environment!")
client = genai.Client(api_key=apikey)
gemma_model = "gemma-3-12b-it"

LLM7_API_KEY = os.getenv("LLM7_API_KEY")
LLM7_BASE_URL = os.getenv("LLM7_BASE_URL", "https://api.llm7.io/v1")
LLM7_MODEL = os.getenv("LLM7_MODEL", "fast")
llm7_client = None
if LLM7_API_KEY:
    try:
        llm7_client = OpenAI(
            base_url=LLM7_BASE_URL,
            api_key=LLM7_API_KEY
        )
    except Exception as e:
        print(f"Error initializing LLM7 client: {e}")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
finnhub_client = None
if FINNHUB_API_KEY:
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Error initializing Finnhub client: {e}")

# Trending stocks API
import requests
from requests.auth import HTTPBasicAuth
import html
APEWISDOM_API_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
ALPACA_MOST_ACTIVE_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=20"
TRENDING_PAGE_SOURCES = ("stocktwits", "reddit", "volume")


def _normalize_trending_page_source(raw_value, default="stocktwits"):
    normalized = (raw_value or "").strip().lower()
    if normalized in TRENDING_PAGE_SOURCES:
        return normalized
    return default


def _get_int_env(var_name, default):
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


BLOG_ARTICLE_FETCH_LIMIT = max(1, _get_int_env("BLOG_ARTICLE_FETCH_LIMIT", 50))


GEMMA_SENTIMENT_TIMEOUT_SECONDS = max(1, _get_int_env("GEMMA_SENTIMENT_TIMEOUT_SECONDS", 8))
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_SENTIMENT_MODEL = os.getenv("CLOUDFLARE_SENTIMENT_MODEL", "@cf/meta/llama-3-8b-instruct")
CLOUDFLARE_TIMEOUT_SECONDS = max(3, _get_int_env("CLOUDFLARE_TIMEOUT_SECONDS", 10))
CLOUDFLARE_BASE_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/"
    if CLOUDFLARE_ACCOUNT_ID else None
)
CLOUDFLARE_ENABLED = bool(CLOUDFLARE_BASE_URL and CLOUDFLARE_API_TOKEN)
_cloudflare_session = requests.Session() if CLOUDFLARE_ENABLED else None
_cloudflare_headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"} if CLOUDFLARE_ENABLED else {}

SENTIMENT_RUN_STATE_TTL_SECONDS = max(60, _get_int_env("SENTIMENT_RUN_STATE_TTL_SECONDS", 300))
_sentiment_run_state = {}
_sentiment_run_lock = threading.Lock()


def _prune_sentiment_run_state_locked(now=None):
    if not _sentiment_run_state:
        return
    now = now or time.time()
    expired = [
        run_id for run_id, meta in _sentiment_run_state.items()
        if now - meta.get('ts', now) > SENTIMENT_RUN_STATE_TTL_SECONDS
    ]
    for run_id in expired:
        _sentiment_run_state.pop(run_id, None)


def _should_force_cloudflare_for_run(run_id):
    if not (run_id and CLOUDFLARE_ENABLED):
        return False
    with _sentiment_run_lock:
        _prune_sentiment_run_state_locked()
        entry = _sentiment_run_state.get(run_id)
        return bool(entry and entry.get('force_cloudflare'))


def _mark_run_force_cloudflare(run_id):
    if not (run_id and CLOUDFLARE_ENABLED):
        return
    with _sentiment_run_lock:
        _prune_sentiment_run_state_locked()
        _sentiment_run_state[run_id] = {
            'ts': time.time(),
            'force_cloudflare': True
        }


AI_RATE_LIMIT_PER_MINUTE = max(0, _get_int_env("GEMMA_MAX_CALLS_PER_MINUTE", 45))
AI_RATE_WINDOW_SECONDS = max(1, _get_int_env("GEMMA_RATE_WINDOW_SECONDS", 60))
_ai_call_timestamps = deque()

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
UNSPLASH_APP_NAME = os.getenv("UNSPLASH_APP_NAME", "stocks-sentiment-analysis")
UNSPLASH_DEFAULT_QUERY = os.getenv("UNSPLASH_DEFAULT_QUERY", "stock market")
UNSPLASH_TIMEOUT_SECONDS = max(3, _get_int_env("UNSPLASH_TIMEOUT_SECONDS", 10))
UNSPLASH_RANDOM_URL = "https://api.unsplash.com/photos/random"
UNSPLASH_ENABLED = bool(UNSPLASH_ACCESS_KEY)
EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN")
MAILGUN_MARKET_LIST_ADDRESS = os.getenv("MAILGUN_MARKET_LIST_ADDRESS", "marketsummary@mail.stocksentimentapp.com")
MAILGUN_FROM_EMAIL = os.getenv("MAILGUN_FROM_EMAIL") or (
    f"Stock Sentiment App <postmaster@{MAILGUN_DOMAIN}>" if MAILGUN_DOMAIN else None
)
MAILGUN_API_BASE = "https://api.mailgun.net/v3"
MAILGUN_MESSAGES_URL = f"{MAILGUN_API_BASE}/{MAILGUN_DOMAIN}/messages" if MAILGUN_DOMAIN else None
MAILGUN_LISTS_BASE_URL = f"{MAILGUN_API_BASE}/lists"
MAILGUN_TIMEOUT_SECONDS = max(5, _get_int_env("MAILGUN_TIMEOUT_SECONDS", 10))
MAILGUN_ENABLED = bool(MAILGUN_API_KEY and MAILGUN_DOMAIN and MAILGUN_MARKET_LIST_ADDRESS and MAILGUN_MESSAGES_URL)


def _build_unsplash_referral_url():
    source_value = (UNSPLASH_APP_NAME or 'stocks-sentiment-analysis').replace(' ', '-').lower()
    return f"https://unsplash.com/?utm_source={source_value}&utm_medium=referral"


def wait_for_ai_rate_slot():
    """Respect a simple in-process rate limit for Gemma API calls."""
    if AI_RATE_LIMIT_PER_MINUTE == 0:
        return

    while True:
        now = time.time()
        window_floor = now - AI_RATE_WINDOW_SECONDS
        while _ai_call_timestamps and _ai_call_timestamps[0] < window_floor:
            _ai_call_timestamps.popleft()

        if len(_ai_call_timestamps) < AI_RATE_LIMIT_PER_MINUTE:
            _ai_call_timestamps.append(now)
            return

        wait_for = AI_RATE_WINDOW_SECONDS - (now - _ai_call_timestamps[0])
        if wait_for <= 0:
            _ai_call_timestamps.popleft()
            continue
        time.sleep(wait_for)


PRICE_CACHE_TTL_SECONDS = max(5, _get_int_env("TRENDING_PRICE_TTL_SECONDS", 120))
_price_snapshot_cache = {}

STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS = max(
    60,
    _get_int_env("STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS", 600)
)
_stocktwits_summary_cache = {}


def _extract_stocktwits_summary_parts(symbol_payload):
    """Return (summary_text, summary_meta, company_name) from a StockTwits payload."""
    trends = symbol_payload.get("trends") or {}
    raw_summary = (
        trends.get("summary")
        or symbol_payload.get("summary")
        or symbol_payload.get("watchlist_description")
        or symbol_payload.get("body")
        or ""
    )
    summary_meta = {
        "source": "stocktwits",
        "summary_at": trends.get("summary_at") or symbol_payload.get("summary_at")
    }
    company_name = symbol_payload.get("title") or symbol_payload.get("symbol") or "Unknown"
    return raw_summary, summary_meta, company_name


def _normalize_stocktwits_summary(summary_payload):
    """Coerce StockTwits summary payloads (string/dict/list) into clean text."""
    if summary_payload is None:
        return None

    if isinstance(summary_payload, str):
        normalized = summary_payload.strip()
        return normalized or None

    if isinstance(summary_payload, dict):
        for key in ("text", "body", "summary", "content"):
            value = summary_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if isinstance(summary_payload, (list, tuple)):
        parts = []
        for part in summary_payload:
            normalized = _normalize_stocktwits_summary(part)
            if normalized:
                parts.append(normalized)
        if parts:
            joined = " ".join(parts).strip()
            return joined or None
        return None

    try:
        normalized = str(summary_payload).strip()
    except Exception:
        return None
    return normalized or None


def get_price_change_snapshot(symbol):
    if not symbol:
        return {"price": None, "change_percent": None}

    normalized = symbol.upper()
    now = time.time()
    cached = _price_snapshot_cache.get(normalized)
    if cached and (now - cached["timestamp"]) < PRICE_CACHE_TTL_SECONDS:
        return cached["data"]

    current_price = None
    previous_close = None

    try:
        ticker = yf.Ticker(normalized)
        fast_info = getattr(ticker, "fast_info", None) or {}
        current_price = (
            fast_info.get("last_price")
            or fast_info.get("lastPrice")
            or fast_info.get("regular_market_price")
            or fast_info.get("regularMarketPrice")
        )
        previous_close = (
            fast_info.get("previous_close")
            or fast_info.get("previousClose")
            or fast_info.get("regular_market_previous_close")
            or fast_info.get("regularMarketPreviousClose")
        )

        if current_price is None or previous_close is None:
            info = ticker.info or {}
            if current_price is None:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if previous_close is None:
                previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    except Exception as exc:
        print(f"Error fetching realtime price for {normalized}: {exc}")

    change_percent = None
    if current_price is not None and previous_close not in (None, 0):
        try:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        except Exception:
            change_percent = None

    snapshot = {"price": current_price, "change_percent": change_percent}
    _price_snapshot_cache[normalized] = {"timestamp": now, "data": snapshot}
    return snapshot


def _parse_retry_after_value(value):
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        retry_dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if retry_dt is None:
        return None

    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=timezone.utc)

    delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def extract_retry_delay_seconds(exc):
    """Extract retry delay from a Gemma API exception, if provided."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            parsed = _parse_retry_after_value(retry_after)
            if parsed is not None:
                return parsed

    message = str(exc).lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|secs|s)", message)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            return None

    return None


def fetch_top_stocks():
    """Fetch top trending Reddit stocks from ApeWisdom API"""
    try:
        resp = requests.get(APEWISDOM_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        stocks_list = data.get("results", [])
        # Defensive: filter out entries with None rank or ticker
        filtered = [s for s in stocks_list if s.get("rank") is not None and s.get("ticker")]
        try:
            top20 = sorted(filtered, key=lambda x: x.get("rank", 999))[:20]
        except Exception as sort_exc:
            print(f"Error sorting trending stocks: {sort_exc}")
            top20 = []
        return top20
    except Exception as e:
        print(f"Error fetching trending stocks: {e}")
        return []

def analyze_trending(top10, include_prices=True):
    """Analyze trending stocks and apply tags based on percent increase and rank changes"""
    results = []
    for rec in top10:
        ticker = rec.get("ticker")
        if not ticker:
            continue
        name = html.unescape(rec.get("name", ""))
        mentions_now = rec.get("mentions", 0)
        mentions_24h = rec.get("mentions_24h_ago", 0)
        rank_24h = rec.get("rank_24h_ago", None)
        rank_now = rec.get("rank", None)

        # Defensive: ensure mentions are ints
        try:
            mentions_now = int(mentions_now) if mentions_now is not None else 0
        except Exception:
            mentions_now = 0
        try:
            mentions_24h = int(mentions_24h) if mentions_24h is not None else 0
        except Exception:
            mentions_24h = 0

        # Defensive: ensure ranks are ints or None
        try:
            rank_24h = int(rank_24h) if rank_24h is not None else None
        except Exception:
            rank_24h = None
        try:
            rank_now = int(rank_now) if rank_now is not None else None
        except Exception:
            rank_now = None

        # percent increase
        if mentions_24h > 0:
            try:
                pct_increase = ((mentions_now - mentions_24h) / mentions_24h) * 100
            except Exception:
                pct_increase = 0
        else:
            pct_increase = 0

        tag = ""
        if mentions_24h > 0 and pct_increase > 50:
            tag = "Trending"
        elif (
            rank_24h is not None and rank_now is not None
            and isinstance(rank_24h, int) and isinstance(rank_now, int)
            and rank_now < rank_24h - 5
        ):
            tag = f"Up {rank_24h - rank_now} Spots"

        price_snapshot = get_price_change_snapshot(ticker) if include_prices else {}
        results.append({
            "ticker": ticker,
            "name": name,
            "pct_increase": pct_increase,
            "tag": tag,
            "price_change_percent": price_snapshot.get("change_percent"),
            "price": price_snapshot.get("price")
        })
    return results


@lru_cache(maxsize=256)
def lookup_company_name(symbol):
    """Resolve ticker to company name via yfinance"""
    if not symbol:
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        return info.get("longName") or info.get("shortName") or info.get("shortName")
    except Exception as e:
        print(f"Error looking up company name for {symbol}: {e}")
        return None


def _cache_stocktwits_summary(symbol, name, summary_text, summary_meta, watchlist_count=None):
    normalized = (symbol or "").upper()
    if not normalized:
        return
    normalized_summary = _normalize_stocktwits_summary(summary_text)
    _stocktwits_summary_cache[normalized] = {
        "ticker": normalized,
        "name": name,
        "summary": normalized_summary,
        "summary_meta": summary_meta or {},
        "watchlist_count": watchlist_count,
        "timestamp": time.time()
    }
    return normalized_summary


def get_stocktwits_summary_entry(symbol):
    normalized = (symbol or "").upper()
    if not normalized:
        return None

    cached = _stocktwits_summary_cache.get(normalized)
    now = time.time()
    if cached and (now - cached.get("timestamp", 0)) < STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS:
        summary_logger.debug(
            "[StockTwits] Serving cached summary for %s (age=%.1fs)",
            normalized,
            now - cached.get("timestamp", 0)
        )
        return cached

    summary_logger.info(
        "[StockTwits] Refreshing summary cache for %s (stale or missing)",
        normalized
    )
    return _refresh_stocktwits_summary(normalized)


def _fetch_stocktwits_symbols(limit=20):
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "darwin",
            "mobile": False
        }
    )
    resp = scraper.get(STOCKTWITS_TRENDING_URL, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    symbols = payload.get("symbols", [])

    filtered = [
        s for s in symbols
        if (s.get("exchange") or "").upper() != "CRYPTO"
        and (s.get("instrument_class") or "").lower() == "stock"
    ]
    return filtered[:limit]


def _refresh_stocktwits_summary(symbol, *, limit=60):
    normalized = (symbol or "").upper()
    if not normalized:
        return None

    try:
        summary_logger.info(
            "[StockTwits] On-demand summary fetch for %s (limit=%d)",
            normalized,
            limit
        )
        symbols = _fetch_stocktwits_symbols(limit=limit)
    except Exception as exc:
        summary_logger.exception(
            "[StockTwits] Error refreshing summary for %s: %s",
            normalized,
            exc
        )
        return None

    for sym in symbols:
        current_symbol = (sym.get("symbol") or "").upper()
        if current_symbol != normalized:
            continue
        raw_summary, summary_meta, company_name = _extract_stocktwits_summary_parts(sym)
        _cache_stocktwits_summary(
            normalized,
            company_name,
            raw_summary,
            summary_meta,
            sym.get("watchlist_count")
        )
        return _stocktwits_summary_cache.get(normalized)

    summary_logger.warning(
        "[StockTwits] Summary refresh miss for %s (not in payload)",
        normalized
    )
    return None


def fetch_stocktwits_trending(limit=20, include_prices=True):
    """Fetch trending symbols from StockTwits"""
    try:
        trimmed = _fetch_stocktwits_symbols(limit=limit)
        results = []
        for idx, sym in enumerate(trimmed, start=1):
            price_info = sym.get("price")
            if isinstance(price_info, dict):
                last_price = price_info.get("last")
                change_pct = price_info.get("change_percent")
            else:
                last_price = sym.get("price")
                change_pct = sym.get("change_percent")

            symbol = (sym.get("symbol") or "").upper()
            resolved_price = last_price
            resolved_change_pct = change_pct
            if include_prices and (resolved_price is None or resolved_change_pct is None):
                snapshot = get_price_change_snapshot(symbol)
                if resolved_price is None:
                    resolved_price = snapshot.get("price")
                if resolved_change_pct is None:
                    resolved_change_pct = snapshot.get("change_percent")

            raw_summary, summary_meta, company_name = _extract_stocktwits_summary_parts(sym)
            cached_summary = None
            has_summary = False
            if idx <= 10:
                cached_summary = _cache_stocktwits_summary(
                    symbol,
                    company_name,
                    raw_summary,
                    summary_meta,
                    sym.get("watchlist_count")
                )
                has_summary = bool(cached_summary)

            results.append({
                "rank": idx,
                "ticker": symbol,
                "name": company_name,
                "watchlist_count": sym.get("watchlist_count"),
                "change_percent": resolved_change_pct,
                "price": resolved_price,
                "price_change_percent": resolved_change_pct,
                "has_summary": has_summary
            })
        return results
    except Exception:
        return []


def fetch_alpaca_most_actives(limit=20, include_prices=True):
    """Fetch most active stocks by volume from Alpaca"""
    headers = {}
    alpaca_key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        headers["Apca-Api-Key-Id"] = alpaca_key
        headers["Apca-Api-Secret-Key"] = alpaca_secret

    try:
        resp = requests.get(ALPACA_MOST_ACTIVE_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        stocks = (
            payload.get("most_actives")
            or payload.get("stocks")
            or payload.get("results")
            or payload.get("data")
            or []
        )

        trimmed = stocks[:limit]
        results = []
        for idx, rec in enumerate(trimmed, start=1):
            change_pct = rec.get("change_percent") or rec.get("percent_change")
            price = rec.get("price") or rec.get("last") or rec.get("close")
            volume = rec.get("volume") or rec.get("volume_1d")
            symbol = (rec.get("symbol") or "").upper()
            resolved_name = rec.get("name") or lookup_company_name(symbol) or symbol or "Unknown"
            if include_prices and (price is None or change_pct is None):
                snapshot = get_price_change_snapshot(symbol)
                if price is None:
                    price = snapshot.get("price")
                if change_pct is None:
                    change_pct = snapshot.get("change_percent")
            results.append({
                "rank": idx,
                "ticker": symbol,
                "name": resolved_name,
                "volume": volume,
                "price": price,
                "change_percent": change_pct,
                "price_change_percent": change_pct
            })
        return results
    except Exception as e:
        print(f"Error fetching Alpaca most actives: {e}")
        return []


MARKET_SUMMARY_INDEXES = (
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq Composite"),
    ("^DJI", "Dow Jones")
)
MARKET_SUMMARY_MAX_HEADLINES = max(3, _get_int_env("MARKET_SUMMARY_MAX_HEADLINES", 8))
MARKET_SUMMARY_RELEASE_HOUR = max(0, min(23, _get_int_env("MARKET_SUMMARY_RELEASE_HOUR", 16)))
MARKET_SUMMARY_RELEASE_MINUTE = max(0, min(59, _get_int_env("MARKET_SUMMARY_RELEASE_MINUTE", 15)))
MARKET_SUMMARY_RETENTION_DAYS = max(30, _get_int_env("MARKET_SUMMARY_RETENTION_DAYS", 90))
MARKET_SUMMARY_ENABLED = os.getenv("ENABLE_MARKET_SUMMARY", "1") not in {"0", "false", "False"}


def _fetch_index_snapshot(symbol, label):
    ticker = yf.Ticker(symbol)
    try:
        hist = ticker.history(period="2d")
    except Exception as exc:
        market_summary_logger.warning("Index history error for %s: %s", symbol, exc)
        hist = None

    close = change = change_pct = None
    if hist is not None and not hist.empty:
        last_row = hist.tail(1)
        prev_row = hist.tail(2).head(1)
        try:
            close = float(last_row["Close"].iloc[0])
        except Exception:
            close = None
        if close is not None and prev_row is not None and not prev_row.empty:
            try:
                prev_close = float(prev_row["Close"].iloc[0])
                change = close - prev_close
                if prev_close:
                    change_pct = (change / prev_close) * 100
            except Exception:
                change = change_pct = None
    return {
        "symbol": symbol,
        "label": label,
        "close": close,
        "change": change,
        "change_percent": change_pct
    }


def get_market_index_snapshots():
    snapshots = []
    for symbol, label in MARKET_SUMMARY_INDEXES:
        snapshot = _fetch_index_snapshot(symbol, label)
        if snapshot:
            snapshots.append(snapshot)
    return snapshots


def get_market_news_digest(max_articles=None):
    limit = max_articles or MARKET_SUMMARY_MAX_HEADLINES
    articles = []
    seen_titles = set()
    try:
        gn = GoogleNews(lang='en', country='US')
    except Exception as exc:
        market_summary_logger.error("Unable to initialize Google News: %s", exc)
        return articles

    queries = [
        "stock market today",
        "wall street wrap",
        "us stocks closing bell"
    ]

    for query in queries:
        try:
            feed = gn.search(query)
        except Exception as exc:
            market_summary_logger.warning("Google News search failed for '%s': %s", query, exc)
            continue

        entries = feed.get('entries') or []
        for entry in entries:
            title = (entry.get('title') or '').strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            summary = entry.get('summary') or entry.get('description') or ''
            link = entry.get('link')
            published_at = entry.get('published') or entry.get('updated')
            source_block = entry.get('source')
            source_title = 'Unknown'
            if isinstance(source_block, dict):
                source_title = source_block.get('title') or source_title
            elif isinstance(source_block, list):
                for item in source_block:
                    if isinstance(item, dict) and item.get('title'):
                        source_title = item['title']
                        break

            articles.append({
                'title': title,
                'description': summary,
                'link': link,
                'publishedAt': published_at,
                'source': source_title
            })

            if len(articles) >= limit:
                return articles

    return articles


def _format_index_line(snapshot):
    close = snapshot.get('close')
    change = snapshot.get('change')
    change_pct = snapshot.get('change_percent')
    if close is None or change is None or change_pct is None:
        return f"{snapshot.get('label')}: data unavailable"
    direction = "up" if change >= 0 else "down"
    return (
        f"{snapshot.get('label')} ({snapshot.get('symbol')}): closed at {close:,.2f}, "
        f"{direction} {abs(change):,.2f} points ({change_pct:+.2f}%)."
    )


def fallback_market_summary_text(summary_date, snapshots, headlines):
    date_str = summary_date.strftime('%A, %B %d, %Y')
    paragraphs = [f"Markets wrap for {date_str}."]
    if snapshots:
        bullet_lines = ' '.join(_format_index_line(s) for s in snapshots)
        paragraphs.append(bullet_lines)
    if headlines:
        headline_text = ' '.join(f"{item['title']} ({item['source']})" for item in headlines[:3])
        paragraphs.append(f"Top stories included {headline_text}.")
    return '\n\n'.join(paragraphs)


def generate_market_summary_text(summary_date, snapshots, headlines):
    if not llm7_client:
        return fallback_market_summary_text(summary_date, snapshots, headlines)

    index_lines = '\n'.join(_format_index_line(s) for s in snapshots) or 'No index data available.'
    news_lines = '\n'.join(
        f"- {item['title']} ({item['source']}): {item['description']}"
        for item in headlines
    ) or 'No major headlines captured.'

    prompt = (
        "Write a concise market wrap article (2-3 short paragraphs) for U.S. equities. "
        "Mention how each major index moved and weave in the headlines when relevant. "
        "Avoid hype, keep it factual, and stay under 180 words.\n\n"
        f"Date: {summary_date.strftime('%A, %B %d, %Y')}\n"
        "Index recap:\n"
        f"{index_lines}\n\n"
        "Headline digest:\n"
        f"{news_lines}\n"
        "Do not use markdown headings."
        "Do not add a title or start with Summary:"
    )

    try:
        response = llm7_client.chat.completions.create(
            model=LLM7_MODEL,
            messages=[
                {"role": "system", "content": "You are a sharp markets reporter."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=350
        )
        content = response.choices[0].message.content.strip()
        if content:
            return content
    except Exception as exc:
        market_summary_logger.error("LLM7 market summary failed: %s", exc)

    return fallback_market_summary_text(summary_date, snapshots, headlines)


def build_market_summary_slug(record):
    if not record:
        return None
    summary_date = getattr(record, 'summary_date', None)
    if summary_date:
        return summary_date.isoformat()
    record_id = getattr(record, 'id', None)
    if record_id:
        return f"id-{record_id}"
    return None


def build_market_summary_permalink(slug):
    if not slug:
        return None
    slug_text = str(slug).strip()
    if not slug_text:
        return None
    return f"/market-summary/{slug_text}"


def build_market_summary_image_payload(record):
    if not (record and UNSPLASH_ENABLED):
        return None

    summary_label = None
    slug = build_market_summary_slug(record)
    summary_locator = slug
    if record.summary_date:
        summary_label = record.summary_date.strftime('%B %d, %Y')
    elif record.published_at and not summary_locator:
        try:
            record_date = record.published_at.date()
        except Exception:
            record_date = None
        if record_date:
            summary_locator = record_date.isoformat()
            summary_label = record_date.strftime('%B %d, %Y')
    if not summary_locator:
        summary_locator = f"id-{record.id or 'latest'}"
    permalink = build_market_summary_permalink(summary_locator)

    article_stub = {
        "link": permalink or f"market-summary://{summary_locator}",
        "title": record.title or "Market Summary",
        "description": record.body or ""
    }
    fallback_query = f"stock market wrap {summary_label}" if summary_label else "stock market wrap"
    try:
        enrich_articles_with_unsplash_images([article_stub], fallback_query)
    except Exception as exc:
        unsplash_logger.error("Market summary image enrichment failed: %s", exc)
        return None
    return article_stub.get('image')


def serialize_market_summary(record):
    if not record:
        return None
    try:
        indices = json.loads(record.index_snapshot or '[]')
    except json.JSONDecodeError:
        indices = []
    try:
        headlines = json.loads(record.headline_sources or '[]')
    except json.JSONDecodeError:
        headlines = []

    def _serialize_datetime(value):
        if not value:
            return None
        if value.tzinfo:
            return value.isoformat()
        return value.replace(tzinfo=timezone.utc).isoformat()

    slug = build_market_summary_slug(record)
    permalink = build_market_summary_permalink(slug)

    return {
        'id': record.id,
        'title': record.title,
        'body': record.body,
        'slug': slug,
        'permalink': permalink,
        'summary_date': record.summary_date.isoformat() if record.summary_date else None,
        'published_at': _serialize_datetime(record.published_at),
        'created_at': _serialize_datetime(record.created_at),
        'updated_at': _serialize_datetime(record.updated_at),
        'indices': indices,
        'headlines': headlines,
        'image': build_market_summary_image_payload(record)
    }


def _parse_iso_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_currency(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    return f"${number:,.2f}"


def _format_signed_percentage(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '—'
    sign = '+' if number >= 0 else ''
    return f"{sign}{number:.2f}%"


def _format_headline_meta(item):
    parts = []
    source = (item.get('source') or '').strip()
    if source:
        parts.append(source)
    stamp = _parse_iso_datetime(item.get('publishedAt'))
    if stamp:
        parts.append(stamp.strftime('%b %d, %Y'))
    return ' • '.join(parts)


def _build_body_html(body):
    if not body:
        return "<p style=\"margin:0;color:#d1d5db;\">No summary available.</p>"
    paragraphs = []
    for chunk in body.split('\n'):
        text = chunk.strip()
        if text:
            paragraphs.append(
                f"<p style=\"margin:0 0 12px;color:#d1d5db;line-height:1.7;\">{html.escape(text)}</p>"
            )
    return ''.join(paragraphs)


def _resolve_summary_datetime(summary):
    if not summary:
        return None
    for field in ('published_at', 'summary_date', 'created_at'):
        candidate = _parse_iso_datetime(summary.get(field))
        if candidate:
            return candidate
    return None


def _format_daily_heading(summary_dt):
    if not summary_dt:
        return "Your Daily Market Summary"
    local = summary_dt.astimezone(EASTERN_TZ)
    month = local.strftime('%B')
    day = local.day
    return f"Your Daily Market Summary: {month} {day}"


def _format_meta_label(summary_dt):
    if not summary_dt:
        return ''
    local = summary_dt.astimezone(EASTERN_TZ)
    return local.strftime('%B %d, %Y • %I:%M %p ET')


def _build_indices_html(indices):
    if not indices:
        return ''
    cards = []
    for payload in indices:
        label = html.escape(payload.get('label') or payload.get('symbol') or 'Index')
        close_label = _format_currency(payload.get('close'))
        try:
            change_pct = float(payload.get('change_percent'))
        except (TypeError, ValueError):
            change_pct = None
        change_label = _format_signed_percentage(change_pct)
        color = '#a1a1aa'
        if change_pct is not None:
            color = '#33ea93' if change_pct >= 0 else '#ef4444'
        cards.append(
            (
                "<div style=\"background:#1c1b27;border-radius:10px;padding:14px;border:1px solid rgba(99,102,241,0.25);\">"
                f"<div style=\"font-size:14px;color:#c4b5fd;margin-bottom:6px;\">{label}</div>"
                f"<div style=\"font-size:20px;font-weight:600;color:#fff;\">{close_label}</div>"
                f"<div style=\"font-size:14px;font-weight:600;color:{color};\">{change_label}</div>"
                "</div>"
            )
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:22px;">'
        + ''.join(cards)
        + '</div>'
    )


def _build_headlines_html(headlines):
    if not headlines:
        return ''
    rows = []
    for item in headlines:
        title = html.escape(item.get('title') or 'Story')
        link = item.get('link') or item.get('url')
        meta = _format_headline_meta(item)
        if link:
            link_html = (
                f"<a href=\"{html.escape(link)}\" style=\"color:#e0e7ff;text-decoration:none;\" target=\"_blank\" rel=\"noopener noreferrer\">{title}</a>"
            )
        else:
            link_html = f"<span style=\"color:#e0e7ff;\">{title}</span>"
        meta_html = html.escape(meta) if meta else ''
        rows.append(
            (
                "<tr>"
                "<td style=\"padding:0 0 10px;vertical-align:top;\">"
                f"<div style=\"font-size:15px;font-weight:600;line-height:1.4;\">{link_html}</div>"
                + (
                    f"<div style=\"font-size:12px;color:#9ca3af;margin-top:2px;text-transform:uppercase;letter-spacing:0.2px;\">{meta_html}</div>"
                    if meta_html
                    else ''
                )
                + "</td>"
                + "</tr>"
            )
        )
    return (
        "<div style=\"background:#13131f;border-radius:10px;padding:18px;border:1px solid rgba(147,51,234,0.15);\">"
        "<h3 style=\"font-size:15px;font-weight:600;color:#c4b5fd;margin:0 0 12px;text-transform:uppercase;letter-spacing:0.3px;\">Headline drivers</h3>"
        "<table role=\"presentation\" style=\"width:100%;border-collapse:collapse;\">"
        + ''.join(rows)
        + "</table></div>"
    )


def _build_image_html(image_payload, title):
    if not image_payload:
        return ''
    image_url = image_payload.get('url') or image_payload.get('image_url')
    if not image_url:
        return ''
    description = html.escape(image_payload.get('description') or f"Market illustration for {title}")
    photographer = (
        image_payload.get('photographer')
        or image_payload.get('photographer_name')
        or 'Unsplash photographer'
    )
    photographer_profile = (
        image_payload.get('photographer_profile')
        or image_payload.get('photographer_profile_url')
        or 'https://unsplash.com'
    )
    unsplash_link = (
        image_payload.get('unsplash_link')
        or image_payload.get('unsplash_photo_link')
        or 'https://unsplash.com'
    )
    return (
        "<div style=\"position:relative;border-radius:12px;overflow:hidden;margin-bottom:20px;border:1px solid rgba(255,255,255,0.05);background:#050505;\">"
        f"<img src=\"{image_url}\" alt=\"{description}\" style=\"width:100%;height:260px;object-fit:cover;display:block;\" />"
        f"<div style=\"position:absolute;right:12px;bottom:12px;background:rgba(0,0,0,0.65);padding:6px 12px;border-radius:999px;font-size:12px;color:#f1f5f9;\">Photo by <a href=\"{photographer_profile}\" style=\"color:#f8fafc;\" target=\"_blank\" rel=\"noopener noreferrer\">{html.escape(photographer)}</a> on <a href=\"{unsplash_link}\" style=\"color:#f8fafc;\" target=\"_blank\" rel=\"noopener noreferrer\">Unsplash</a></div>"
        "</div>"
    )


def build_market_summary_email_html(summary, heading, summary_dt):
    body_html = _build_body_html(summary.get('body'))
    indices_html = _build_indices_html(summary.get('indices'))
    headlines_html = _build_headlines_html(summary.get('headlines'))
    image_html = _build_image_html(summary.get('image'), heading)
    meta_label = _format_meta_label(summary_dt)
    return (
        "<div style=\"background:#000;padding:32px 20px;font-family:'Inter','Segoe UI',sans-serif;color:#fff;\">"
        "<div style=\"max-width:640px;margin:0 auto;background:#0a0a0a;border-radius:16px;padding:32px;border:1px solid rgba(147,51,234,0.2);box-shadow:0 20px 45px rgba(0,0,0,0.35);\">"
        "<div style=\"text-transform:uppercase;font-size:13px;letter-spacing:0.35px;color:#a78bfa;margin-bottom:8px;\">Daily Market Summary</div>"
        f"<h1 style=\"margin:0 0 6px;font-size:26px;line-height:1.2;color:#fff;\">{html.escape(heading)}</h1>"
        f"<div style=\"font-size:13px;color:#a1a1aa;margin-bottom:18px;letter-spacing:0.3px;text-transform:uppercase;\">{html.escape(meta_label)}</div>"
        f"{image_html}"
        "<div style=\"background:#11111a;border:1px solid rgba(147,51,234,0.15);border-radius:12px;padding:24px;margin-bottom:24px;\">"
        f"{body_html}"
        "</div>"
        f"{indices_html}"
        f"{headlines_html}"
        "<div style=\"margin-top:28px;text-align:center;font-size:12px;color:#9ca3af;\">"
        "<a href=\"%unsubscribe_url%\" style=\"color:#c4b5fd;text-decoration:underline;\">Unsubscribe</a>"
        "</div>"
        "</div>"
        "</div>"
    )


def _is_valid_email_address(value):
    if not value:
        return False
    return bool(EMAIL_REGEX.match(value))


def add_member_to_mailgun_list(email):
    if not (MAILGUN_ENABLED and MAILGUN_API_KEY and MAILGUN_LISTS_BASE_URL and MAILGUN_MARKET_LIST_ADDRESS):
        raise RuntimeError("Mailgun list integration is unavailable.")
    url = f"{MAILGUN_LISTS_BASE_URL}/{MAILGUN_MARKET_LIST_ADDRESS}/members"
    response = requests.post(
        url,
        auth=HTTPBasicAuth("api", MAILGUN_API_KEY),
        data={
            "address": email,
            "subscribed": "yes",
            "upsert": "yes"
        },
        timeout=MAILGUN_TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Mailgun list update failed ({response.status_code})")


def dispatch_market_summary_email(summary_record):
    if not (summary_record and MAILGUN_ENABLED and MAILGUN_FROM_EMAIL and MAILGUN_MESSAGES_URL):
        raise RuntimeError("Mailgun is not fully configured.")
    serialized = serialize_market_summary(summary_record)
    summary_dt = _resolve_summary_datetime(serialized)
    heading = _format_daily_heading(summary_dt)
    html_body = build_market_summary_email_html(serialized, heading, summary_dt)
    text_body = serialized.get('body') or heading
    to_header = f"Market Summary Subscribers <{MAILGUN_MARKET_LIST_ADDRESS}>"
    response = requests.post(
        MAILGUN_MESSAGES_URL,
        auth=HTTPBasicAuth("api", MAILGUN_API_KEY),
        data={
            "from": MAILGUN_FROM_EMAIL,
            "to": to_header,
            "subject": heading,
            "text": text_body,
            "html": html_body
        },
        timeout=MAILGUN_TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        snippet = response.text[:200] if response.text else ''
        raise RuntimeError(f"Mailgun broadcast failed ({response.status_code}) {snippet}")
    return True


def get_latest_market_summary_record():
    return (
        MarketSummary.query
        .order_by(MarketSummary.summary_date.desc(), MarketSummary.published_at.desc(), MarketSummary.id.desc())
        .first()
    )


def resolve_market_summary_by_slug(summary_slug):
    if not summary_slug:
        return None
    slug_text = str(summary_slug).strip()
    if not slug_text:
        return None

    if slug_text.startswith('id-'):
        try:
            record_id = int(slug_text.split('id-', 1)[1])
        except ValueError:
            record_id = None
        if record_id:
            record = db.session.get(MarketSummary, record_id)
            if record:
                return record

    try:
        summary_date = datetime.strptime(slug_text, '%Y-%m-%d').date()
    except ValueError:
        summary_date = None
    if summary_date:
        return MarketSummary.query.filter_by(summary_date=summary_date).first()

    return None


def send_market_summary_to_recipient(summary_record, recipient_email):
    if not (summary_record and recipient_email and MAILGUN_ENABLED and MAILGUN_FROM_EMAIL and MAILGUN_MESSAGES_URL):
        raise RuntimeError("Unable to send market summary to recipient.")

    serialized = serialize_market_summary(summary_record)
    summary_dt = _resolve_summary_datetime(serialized)
    heading = _format_daily_heading(summary_dt)
    html_body = build_market_summary_email_html(serialized, heading, summary_dt)
    text_body = serialized.get('body') or heading

    response = requests.post(
        MAILGUN_MESSAGES_URL,
        auth=HTTPBasicAuth("api", MAILGUN_API_KEY),
        data={
            "from": MAILGUN_FROM_EMAIL,
            "to": recipient_email,
            "subject": heading,
            "text": text_body,
            "html": html_body
        },
        timeout=MAILGUN_TIMEOUT_SECONDS
    )
    if response.status_code >= 400:
        snippet = response.text[:200] if response.text else ''
        raise RuntimeError(f"Mailgun direct send failed ({response.status_code}) {snippet}")
    return True


def ensure_market_summary_email_sent(summary_record):
    if not summary_record or not MAILGUN_ENABLED:
        return False

    log_entry = MarketSummarySendLog.query.filter_by(summary_id=summary_record.id).first()
    if log_entry and log_entry.status == 'sent':
        return False
    if not log_entry:
        log_entry = MarketSummarySendLog(summary_id=summary_record.id)
        db.session.add(log_entry)

    try:
        dispatch_market_summary_email(summary_record)
        log_entry.status = 'sent'
        log_entry.sent_at = _utcnow_naive()
        log_entry.error_message = None
        db.session.commit()
        market_summary_logger.info(
            "Market summary email dispatched for summary_id=%s",
            summary_record.id
        )
        return True
    except Exception as exc:
        db.session.rollback()
        log_entry.status = 'failed'
        log_entry.error_message = str(exc)[:250]
        db.session.add(log_entry)
        db.session.commit()
        market_summary_logger.error("Failed to send market summary email: %s", exc)
        return False

def prune_market_summary_history(cutoff_date=None):
    if not MARKET_SUMMARY_ENABLED:
        return 0

    if cutoff_date is None:
        cutoff_date = datetime.now(EASTERN_TZ).date() - timedelta(days=MARKET_SUMMARY_RETENTION_DAYS)

    if cutoff_date > datetime.now(EASTERN_TZ).date():
        return 0

    deleted = MarketSummary.query.filter(MarketSummary.summary_date < cutoff_date).delete()
    if deleted:
        db.session.commit()
    return deleted


def upsert_market_summary_sitemap_entry(summary_date):
    """Ensure sitemap.xml lists the market-summary detail page for the given date."""
    if not summary_date:
        return False

    sitemap_path = SITEMAP_FILE_PATH
    if not sitemap_path.exists():
        return False

    lastmod_text = summary_date.strftime('%Y-%m-%d')
    target_loc = f"{MARKET_SUMMARY_SITEMAP_BASE_URL}/{lastmod_text}"
    stock_market_today_loc = f"{MARKET_SUMMARY_SITEMAP_BASE_URL}/stock-market-today"

    try:
        tree = ET.parse(sitemap_path)
    except ET.ParseError as exc:
        raise RuntimeError(f"Sitemap parsing failed: {exc}") from exc

    root = tree.getroot()
    ns_uri = SITEMAP_XML_NAMESPACE
    def _ensure_url_lastmod(loc_value, lastmod_value):
        url_node = None
        for node in root.findall(f"{{{ns_uri}}}url"):
            loc_node = node.find(f"{{{ns_uri}}}loc")
            if loc_node is not None and (loc_node.text or "").strip() == loc_value:
                url_node = node
                break

        if url_node is None:
            url_node = ET.SubElement(root, f"{{{ns_uri}}}url")
            loc_node = ET.SubElement(url_node, f"{{{ns_uri}}}loc")
            loc_node.text = loc_value
        else:
            loc_node = url_node.find(f"{{{ns_uri}}}loc")
            if loc_node is None:
                loc_node = ET.SubElement(url_node, f"{{{ns_uri}}}loc")
            loc_node.text = loc_value

        lastmod_node = url_node.find(f"{{{ns_uri}}}lastmod")
        if lastmod_node is None:
            lastmod_node = ET.SubElement(url_node, f"{{{ns_uri}}}lastmod")
        lastmod_node.text = lastmod_value

    # Individual market summary article URL (date-based).
    _ensure_url_lastmod(target_loc, lastmod_text)
    # SEO landing page that always shows the latest summary.
    _ensure_url_lastmod(stock_market_today_loc, lastmod_text)

    tmp_path = sitemap_path.with_name(sitemap_path.name + ".tmp")
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    try:
        tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
        tmp_path.replace(sitemap_path)
    except OSError as exc:
        raise RuntimeError(f"Sitemap write failed: {exc}") from exc

    return True


def ensure_market_summary_for_date(target_date=None, force=False):
    if not MARKET_SUMMARY_ENABLED:
        return None

    now_et = datetime.now(EASTERN_TZ)
    summary_date = target_date or now_et.date()
    if not force and summary_date.weekday() >= 5:
        return None

    dedupe_market_summaries(summary_date)
    existing = MarketSummary.query.filter_by(summary_date=summary_date).first()
    if existing and not force:
        try:
            upsert_market_summary_sitemap_entry(existing.summary_date)
        except Exception as exc:
            market_summary_logger.error(
                "Failed to sync sitemap entry for %s: %s",
                existing.summary_date,
                exc
            )
        return existing

    snapshots = get_market_index_snapshots()
    headlines = get_market_news_digest()
    if not snapshots and not headlines:
        raise RuntimeError("Insufficient data for market summary")

    body = generate_market_summary_text(summary_date, snapshots, headlines)
    title = f"Market Summary — {summary_date.strftime('%B %d, %Y')}"
    timestamp = _utcnow_naive()

    index_payload = json.dumps(snapshots)
    headline_payload = json.dumps(headlines)

    def _apply_payload(record):
        record.title = title
        record.body = body
        record.published_at = timestamp
        record.index_snapshot = index_payload
        record.headline_sources = headline_payload

    try:
        target_record = existing or MarketSummary(summary_date=summary_date)
        if not existing:
            db.session.add(target_record)
        _apply_payload(target_record)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        dedupe_market_summaries(summary_date)
        conflict_record = MarketSummary.query.filter_by(summary_date=summary_date).first()
        if conflict_record and not force:
            return conflict_record
        if not conflict_record:
            conflict_record = MarketSummary(summary_date=summary_date)
            db.session.add(conflict_record)
        _apply_payload(conflict_record)
        try:
            db.session.commit()
            target_record = conflict_record
        except Exception as exc:
            db.session.rollback()
            market_summary_logger.error(
                "Failed to persist market summary after dedupe: %s", exc
            )
            raise
    except Exception as exc:
        db.session.rollback()
        market_summary_logger.error("Failed to persist market summary: %s", exc)
        raise

    try:
        upsert_market_summary_sitemap_entry(target_record.summary_date)
    except Exception as exc:
        market_summary_logger.error(
            "Failed to update sitemap for %s: %s",
            target_record.summary_date,
            exc
        )

    try:
        pruned = prune_market_summary_history()
        if pruned:
            market_summary_logger.info("Pruned %s stale market summaries", pruned)
    except Exception as exc:
        market_summary_logger.error("Failed to prune market summary history: %s", exc)

    return target_record


def run_market_summary_job():
    with app.app_context():
        try:
            summary = ensure_market_summary_for_date()
            if summary:
                market_summary_logger.info(
                    "Market summary refreshed for %s", summary.summary_date
                )
                ensure_market_summary_email_sent(summary)
        except Exception as exc:
            market_summary_logger.error("Market summary job failed: %s", exc)


market_summary_scheduler = None
_scheduler_started = False
_runtime_bootstrap_lock = threading.Lock()
_runtime_bootstrap_complete = False


def _should_start_market_scheduler():
    if not MARKET_SUMMARY_ENABLED:
        return False
    if os.getenv("FLASK_SKIP_SCHEDULER") == "1":
        return False
    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        return False
    return True


def start_market_summary_scheduler():
    global market_summary_scheduler, _scheduler_started
    if _scheduler_started or not _should_start_market_scheduler():
        return
    if market_summary_scheduler and market_summary_scheduler.running:
        return
    market_summary_scheduler = BackgroundScheduler(timezone=EASTERN_TZ)
    trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=MARKET_SUMMARY_RELEASE_HOUR,
        minute=MARKET_SUMMARY_RELEASE_MINUTE,
        timezone=EASTERN_TZ
    )
    market_summary_scheduler.add_job(
        run_market_summary_job,
        trigger=trigger,
        id="market_summary_job",
        replace_existing=True
    )
    market_summary_scheduler.start()
    atexit.register(lambda: market_summary_scheduler.shutdown(wait=False))
    _scheduler_started = True


def bootstrap_market_summary_if_needed():
    if not MARKET_SUMMARY_ENABLED:
        return
    now_et = datetime.now(EASTERN_TZ)
    if now_et.weekday() >= 5:
        return
    minutes_now = now_et.hour * 60 + now_et.minute
    release_minutes = MARKET_SUMMARY_RELEASE_HOUR * 60 + MARKET_SUMMARY_RELEASE_MINUTE
    if minutes_now < release_minutes:
        return
    existing = MarketSummary.query.filter_by(summary_date=now_et.date()).first()
    if existing:
        return
    try:
        ensure_market_summary_for_date()
        market_summary_logger.info("Bootstrapped market summary for %s", now_et.date())
    except Exception as exc:
        market_summary_logger.error("Bootstrap market summary failed: %s", exc)


def ensure_market_runtime_bootstrap():
    """Initialize market summary data and scheduler once per process."""
    global _runtime_bootstrap_complete
    if _runtime_bootstrap_complete:
        return

    with _runtime_bootstrap_lock:
        if _runtime_bootstrap_complete:
            return
        with app.app_context():
            bootstrap_market_summary_if_needed()
        start_market_summary_scheduler()
        _runtime_bootstrap_complete = True


@app.before_request
def _ensure_market_runtime_bootstrap():
    ensure_market_runtime_bootstrap()

def get_trending_source_data(source, include_prices=True):
    """Return trending data for a specific source identifier."""
    normalized = (source or "").lower()

    if normalized == "stocktwits":
        return fetch_stocktwits_trending(include_prices=include_prices)
    if normalized == "reddit":
        try:
            return analyze_trending(fetch_top_stocks(), include_prices=include_prices)
        except Exception as e:
            print(f"Error in reddit trending: {e}")
            return []
    if normalized == "volume":
        return fetch_alpaca_most_actives(include_prices=include_prices)

    return None


def _should_include_prices(arg_value):
    if arg_value is None:
        return True
    normalized = str(arg_value).strip().lower()
    return normalized not in {"0", "false", "no"}


def get_stock_info(symbol):
    """Get stock information using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', 0)

        # Calculate change
        change = current_price - previous_close if previous_close else 0
        change_percent = (change / previous_close * 100) if previous_close else 0

        # Extract the year founded from the business summary using regex
        description = info.get('longBusinessSummary', '')
        year_founded = 'N/A'
        if description:
            match = re.search(r"in (\b(?:19|20)\d{2}\b)", description)
            if match:
                year_founded = int(match.group(1))
        #Get company officers to find CEO
        ceo_name = 'N/A'
        officers = info.get('companyOfficers', [])
        for officer in officers:
            title = officer.get('title', '').lower()
            if 'ceo' in title:
                ceo_name = officer.get('name', 'N/A')
                break
        return {
            'symbol': symbol,
            'price': current_price,
            'change': change,
            'changePercent': change_percent,
            'afterHoursPrice': info.get('postMarketPrice'),
            'afterHoursChange': info.get('postMarketChange'),
            'afterHoursChangePercent': info.get('postMarketChangePercent'),
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('forwardPE') or info.get('trailingPE', 0),
            'dividendYield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
            'avgVolume': info.get('averageDailyVolume10Day', 0),
            'dayHigh': info.get('dayHigh', 0),
            'dayLow': info.get('dayLow', 0),
            'openPrice': info.get('open') or info.get('regularMarketOpen', 0),
            'volume': info.get('volume', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
            'companyName': info.get('longName') or info.get('shortName', symbol),
            'ceo': ceo_name,
            'employees': info.get('fullTimeEmployees'),
            'city': info.get('city'),
            'state': info.get('state'),
            'country': info.get('country'),
            'industry': info.get('industry'),
            'sector': info.get('sector'),
            'website': info.get('website'),
            'description': description,
            'yearFounded': year_founded
        }
    except Exception as e:
        print(f"Error getting stock info: {e}")
        return None


def get_historical_data(symbol, period='1d'):
    """Get historical price data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)

        if period == '1d':
            hist = ticker.history(period='5d', interval='5m')
            if not hist.empty:
                last_date = hist.index[-1].date()
                hist = hist[hist.index.date == last_date]

        elif period in ['7d', '1w']:
            end = datetime.now()
            start = end - timedelta(days=7)
            hist = yf.download(
                symbol,
                start=start,
                end=end,
                interval='1h',
                prepost=True,
                progress=False
            )

        else:
            interval = None
            if period == '1mo':
                interval = '90m'

            if interval:
                hist = ticker.history(period=period, interval=interval)
            else:
                hist = ticker.history(period=period)

        if hist.empty:
            return []

        data = [
            {"date": idx.strftime("%Y-%m-%d %H:%M:%S"), "price": float(row["Close"])}
            for idx, row in hist.iterrows()
        ]
        return data

    except Exception as e:
        print(f"Error getting historical data: {e}")
        return []


def get_news_articles(stock_symbol, num_articles=10):
    """Get news articles using Google News"""
    try:
        gn = GoogleNews(lang='en', country='US')
        search_result = gn.search(stock_symbol + " stock")
        entries = search_result.get('entries', [])

        articles = []
        for entry in entries[:num_articles]:
            articles.append({
                'title': entry.get('title', 'No title'),
                'description': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'publishedAt': entry.get('published', ''),
                'source': entry.get('source', {}).get('title', 'Unknown')
            })
        return articles
    except Exception as e:
        print(f"Error getting news: {e}")
        return []


def _build_article_locator(article):
    if not article:
        return None
    for key in ('link', 'url'):
        candidate = (article.get(key) or '').strip()
        if candidate:
            return candidate
    title = (article.get('title') or '').strip()
    if title:
        return title
    return None


def _fingerprint_article(article):
    locator = _build_article_locator(article)
    if not locator:
        return None
    try:
        return hashlib.sha256(locator.encode('utf-8')).hexdigest()
    except Exception:
        return None


def _sanitize_unsplash_query_text(value):
    if not value:
        return None
    cleaned = re.sub(r'[^A-Za-z0-9\s]', ' ', value)
    cleaned = ' '.join(cleaned.split())
    if not cleaned:
        return None
    return ' '.join(cleaned.split()[:6])


def _resolve_unsplash_query(article, fallback_query=None):
    return _sanitize_unsplash_query_text(UNSPLASH_DEFAULT_QUERY) or UNSPLASH_DEFAULT_QUERY


def _unsplash_headers():
    if not UNSPLASH_ACCESS_KEY:
        return {}
    return {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}


def _append_utm_params(url):
    if not url:
        return url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    source_value = (UNSPLASH_APP_NAME or 'stocks-sentiment-analysis').replace(' ', '-').lower()
    query_items['utm_source'] = source_value
    query_items['utm_medium'] = 'referral'
    new_query = urlencode(query_items)
    new_parts = parsed._replace(query=new_query)
    return urlunparse(new_parts)


def _request_unsplash_photo(query):
    if not UNSPLASH_ENABLED:
        return None
    params = {
        'query': query or UNSPLASH_DEFAULT_QUERY,
        'orientation': 'landscape',
        'count': 1,
        'content_filter': 'high'
    }
    try:
        resp = requests.get(
            UNSPLASH_RANDOM_URL,
            params=params,
            headers=_unsplash_headers(),
            timeout=UNSPLASH_TIMEOUT_SECONDS
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, list):
            return payload[0] if payload else None
        return payload
    except Exception as exc:
        unsplash_logger.warning("Unsplash lookup failed for '%s': %s", query, exc)
        return None


def _register_unsplash_download(download_url):
    if not (UNSPLASH_ENABLED and download_url):
        return
    try:
        requests.get(
            download_url,
            params={'client_id': UNSPLASH_ACCESS_KEY},
            headers=_unsplash_headers(),
            timeout=UNSPLASH_TIMEOUT_SECONDS
        )
    except Exception as exc:
        unsplash_logger.debug("Unsplash download ping failed: %s", exc)


def _persist_article_image(article_key, article_url, query, photo):
    if not photo:
        return None
    urls = photo.get('urls') or {}
    image_url = urls.get('regular') or urls.get('full')
    if not image_url:
        return None
    user_info = photo.get('user') or {}
    user_links = user_info.get('links') or {}
    photo_links = photo.get('links') or {}

    profile_link = user_links.get('html')
    if not profile_link and user_info.get('username'):
        profile_link = f"https://unsplash.com/@{user_info.get('username')}"
    photo_link = photo_links.get('html')
    if not photo_link and photo.get('id'):
        photo_link = f"https://unsplash.com/photos/{photo.get('id')}"

    record = ArticleImage(
        article_key=article_key,
        article_url=article_url,
        query=query,
        image_url=image_url,
        thumbnail_url=urls.get('small') or urls.get('thumb'),
        description=photo.get('description') or photo.get('alt_description'),
        photographer_name=user_info.get('name'),
        photographer_username=user_info.get('username'),
        photographer_profile_url=_append_utm_params(profile_link or _build_unsplash_referral_url()),
        unsplash_photo_id=photo.get('id'),
        unsplash_photo_link=_append_utm_params(photo_link or _build_unsplash_referral_url())
    )

    db.session.add(record)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        existing = (
            db.session.query(ArticleImage)
            .filter_by(article_key=article_key)
            .first()
        )
        return existing.as_payload() if existing else None
    except Exception as exc:
        db.session.rollback()
        unsplash_logger.error("Unable to persist article image: %s", exc)
        return None

    return record.as_payload()


def fetch_article_image_payload(article, fallback_query=None):
    if not UNSPLASH_ENABLED:
        return None
    article_key = _fingerprint_article(article)
    if not article_key:
        return None

    existing = (
        db.session.query(ArticleImage)
        .filter_by(article_key=article_key)
        .first()
    )
    if existing:
        return existing.as_payload()

    query = _resolve_unsplash_query(article, fallback_query)
    photo = _request_unsplash_photo(query)
    if not photo:
        return None

    download_url = (photo.get('links') or {}).get('download_location')
    _register_unsplash_download(download_url)

    return _persist_article_image(
        article_key,
        _build_article_locator(article),
        query,
        photo
    )


def enrich_articles_with_unsplash_images(articles, fallback_query=None):
    if not (UNSPLASH_ENABLED and articles):
        return
    for article in articles:
        if article.get('image'):
            continue
        image_payload = fetch_article_image_payload(article, fallback_query)
        if image_payload:
            article['image'] = image_payload


def fetch_finnhub_company_news(symbol, max_articles=6, lookback_days=5):
    if not finnhub_client or not symbol:
        return []

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        raw_articles = finnhub_client.company_news(
            symbol,
            _from=start_date.isoformat(),
            to=end_date.isoformat()
        ) or []
    except Exception as e:
        print(f"Error fetching Finnhub news for {symbol}: {e}")
        return []

    sanitized = []
    seen = set()
    for article in raw_articles:
        ts = article.get("datetime")
        iso_time = None
        if isinstance(ts, (int, float)):
            try:
                iso_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            except Exception:
                iso_time = None

        headline = (article.get("headline") or "").strip()
        url = (article.get("url") or "").strip()
        if not headline:
            continue
        dedupe_key = (headline.lower(), url.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        sanitized.append({
            "headline": headline,
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "url": url,
            "publishedAt": iso_time,
            "datetime": ts
        })

    sanitized.sort(key=lambda item: item.get("datetime") or 0, reverse=True)
    trimmed = [s for s in sanitized if s["headline"]][:max_articles]
    return trimmed


def convert_google_news_for_movement(symbol, max_articles=6):
    google_articles = get_news_articles(symbol, max_articles)
    normalized = []
    for entry in google_articles:
        headline = (entry.get('title') or '').strip()
        if not headline:
            continue
        normalized.append({
            "headline": headline,
            "summary": entry.get('description', ''),
            "source": entry.get('source', 'Google News'),
            "url": entry.get('link'),
            "publishedAt": entry.get('publishedAt') or entry.get('published')
        })
    return normalized


def summarize_stock_movement(symbol, company_name, change_percent, articles):
    if not llm7_client or not articles:
        return None

    bullet_lines = []
    for idx, article in enumerate(articles, start=1):
        published = article.get("publishedAt") or "Unknown time"
        summary = article.get("summary") or ""
        bullet_lines.append(
            f"{idx}. {article.get('headline')} ({article.get('source')}) on {published}: {summary}"
        )

    change_direction = "up" if change_percent >= 0 else "down"
    change_abs = abs(change_percent)
    prompt = (
        f"Company: {company_name} ({symbol})\n"
        f"Intraday move: {change_direction} {change_abs:.2f}% today.\n\n"
        "Recent catalysts:\n"
        + "\n".join(bullet_lines)
        + "\n\n"
          "Write a concise 2-3 sentence summary explaining the most likely reasons for the move. "
          "Blend the headlines into a cohesive narrative and mention key drivers."
          "Try to incorporate quantitative details where possible."
          "If the articles do not explain the move, clearly state that."
          "Avoid speculation and do not invent any facts."
          "Do not include a headline or summary section indicators, just give the summary."
          "Do not specifically mention the stock price or percentage change in the summary."
          "Do not mention any specific articles."
          "Refer to the company by its ticker symbol rather than by name"
          "Do not directly refer to what the articles say or do not say."
    )

    try:
        response = llm7_client.chat.completions.create(
            model=LLM7_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sharp markets reporter who explains price action using headlines."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=220
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error generating movement summary: {e}")
        return None


def fallback_movement_summary(company_name, change_percent, articles, source_label):
    direction = "higher" if change_percent >= 0 else "lower"
    base = f"{company_name} is trading {direction} by roughly {abs(change_percent):.2f}% today."

    if articles:
        highlights = [article.get('headline') for article in articles[:3] if article.get('headline')]
        if highlights:
            joined = "; ".join(highlights)
            return f"{base} Recent {source_label} headlines include {joined}."

    return f"{base} No fresh {source_label} headlines surfaced yet, so keep an eye on upcoming catalysts or filings."


def build_movement_insight(stock_info):
    change_percent = stock_info.get('changePercent')
    if change_percent is None or abs(change_percent) < 3:
        return None

    symbol = stock_info.get('symbol')
    company_name = stock_info.get('companyName') or symbol
    articles = fetch_finnhub_company_news(symbol)
    article_source_label = "Finnhub"
    if not articles:
        articles = convert_google_news_for_movement(symbol)
        article_source_label = "news"

    summary = summarize_stock_movement(symbol, company_name, change_percent, articles)
    if not summary:
        summary = fallback_movement_summary(company_name, change_percent, articles, article_source_label)
    if not summary:
        summary = f"{company_name} is moving sharply today, but no catalysts were found."

    sources = []
    for article in articles:
        if not article.get('headline'):
            continue
        sources.append({
            "headline": article.get('headline'),
            "source": article.get('source'),
            "url": article.get('url'),
            "publishedAt": article.get('publishedAt')
        })

    return {
        "summary": summary,
        "changePercent": change_percent,
        "sources": sources
    }


def _build_sentiment_prompt(company_name, article_title, article_description):
    return (
        f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
        f"to the company {company_name}.\n\n"
        f"Title: {article_title}\n"
        f"Description: {article_description}\n\n"
        f"Do not assume anything not explicitly stated in the title or description.\n"
        f"Respond with only one word and nothing else: positive, negative, or neutral."
    )


def _is_model_overloaded_error(exc):
    if hasattr(exc, 'args') and exc.args and isinstance(exc.args[0], dict):
        err = exc.args[0]
        if (
            isinstance(err, dict)
            and 'error' in err
            and err['error'].get('code') == 503
            and 'model is overloaded' in err['error'].get('message', '').lower()
        ):
            return True
    return False


def _extract_sentiment_label(raw_text):
    cleaned = (raw_text or '').strip()

    def _normalize(value):
        if value is None:
            return None
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                return None
        lowered = value.strip().lower()
        lowered = lowered.strip(" .,!?:;'\"[]{}()")
        if lowered in {'positive', 'negative', 'neutral'}:
            return lowered
        for target in ('positive', 'negative', 'neutral'):
            if re.search(rf"\b{target}\b", lowered):
                return target
        return None

    if not cleaned:
        return None

    if cleaned[0] in {'{', '['}:
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            app_logger.warning(
                "Gemma PI returned invalid JSON, defaulting to neutral (error=%s, snippet=%s)",
                exc,
                cleaned[:120]
            )
            return None

        if isinstance(parsed, dict):
            for key in ('sentiment', 'label', 'value'):
                normalized = _normalize(parsed.get(key))
                if normalized:
                    return normalized
            return None

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    for key in ('sentiment', 'label', 'value'):
                        normalized = _normalize(item.get(key))
                        if normalized:
                            return normalized
                else:
                    normalized = _normalize(item)
                    if normalized:
                        return normalized
            return None

    return _normalize(cleaned)


def analyze_sentiment_cloudflare(article_title, article_description, company_name):
    if not CLOUDFLARE_ENABLED:
        return None

    prompt = _build_sentiment_prompt(company_name, article_title, article_description)
    payload = {
        "messages": [
            {"role": "system", "content": "You are a stock expert"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = _cloudflare_session.post(
            f"{CLOUDFLARE_BASE_URL}{CLOUDFLARE_SENTIMENT_MODEL}",
            headers=_cloudflare_headers,
            json=payload,
            timeout=CLOUDFLARE_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        app_logger.warning("Cloudflare sentiment request failed: %s", exc)
        return None

    result_payload = data.get('result') or {}
    raw_text = result_payload.get('response') or result_payload.get('output') or ''
    sentiment = _extract_sentiment_label(raw_text)
    if sentiment:
        app_logger.info("[Cloudflare AI] Sentiment resolved via fallback for %s", company_name)
    return sentiment


def analyze_sentiment_gemma(article_title, article_description, company_name, run_id=None):
    """Analyze sentiment using Google Gemma AI with Cloudflare fallback for slow/error cases."""
    if _should_force_cloudflare_for_run(run_id):
        fallback = analyze_sentiment_cloudflare(article_title, article_description, company_name)
        if fallback:
            app_logger.info(
                "Using Cloudflare sentiment due to prior fallback for run %s (company=%s)",
                run_id,
                company_name
            )
            return fallback
        app_logger.warning(
            "Cloudflare forced fallback did not return a sentiment for run %s", run_id
        )
        return 'neutral'
    prompt = _build_sentiment_prompt(company_name, article_title, article_description)

    fallback_reason = None
    last_exception = None
    gemma_result = None

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            wait_for_ai_rate_slot()
            request_started = time.perf_counter()
            response = client.models.generate_content(
                model=gemma_model,
                contents=prompt
            )
            elapsed = time.perf_counter() - request_started

            raw_text = getattr(response, 'text', '')
            app_logger.info(
                "[Gemma] Sentiment raw (attempt=%s, company=%s): %s",
                attempt + 1,
                company_name,
                str(raw_text)[:240]
            )
            sentiment = _extract_sentiment_label(raw_text) or 'neutral'
            gemma_result = sentiment
            if elapsed > GEMMA_SENTIMENT_TIMEOUT_SECONDS:
                fallback_reason = f"slow_response_{elapsed:.2f}s"
                break
            return sentiment
        except Exception as exc:
            last_exception = exc
            if _is_model_overloaded_error(exc):
                fallback_reason = 'model_overloaded'
                break
            retry_delay = extract_retry_delay_seconds(exc)
            if retry_delay:
                print(f"Gemma quota hit, waiting {retry_delay:.2f}s before retrying...")
                time.sleep(retry_delay)
                continue
            print(f"Error analyzing sentiment: {exc}")
            fallback_reason = fallback_reason or 'api_error'
            break

    if fallback_reason and CLOUDFLARE_ENABLED:
        _mark_run_force_cloudflare(run_id)
        fallback = analyze_sentiment_cloudflare(article_title, article_description, company_name)
        if fallback:
            app_logger.info("Using Cloudflare sentiment fallback (reason=%s)", fallback_reason)
            return fallback

    if gemma_result is not None and not fallback_reason:
        return gemma_result

    if gemma_result is not None:
        return gemma_result

    if last_exception and _is_model_overloaded_error(last_exception):
        raise RuntimeError('MODEL_OVERLOADED')

    return 'neutral'


def build_sentiment_payload(symbol, company_name=None, *, sentiment_run_id=None):
    articles = get_news_articles(symbol, 10)
    fallback_query = f"{company_name or symbol} stock market"
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    run_id = str(sentiment_run_id).strip() if sentiment_run_id else f"bulk-{uuid.uuid4().hex}"
    for article in articles:
        sentiment = analyze_sentiment_gemma(
            article['title'],
            article['description'],
            company_name or symbol,
            run_id=run_id
        )
        article['sentiment'] = sentiment
        if sentiment in sentiment_summary:
            sentiment_summary[sentiment] += 1

    if all(value == 0 for value in sentiment_summary.values()):
        overall_sentiment = 'neutral'
    else:
        overall_sentiment = max(sentiment_summary, key=sentiment_summary.get)

    return articles, sentiment_summary, overall_sentiment


@app.route('/')
@limiter.limit("50 per minute")
def index():
    '''Render the main page'''
    return render_template(
        'index.html',
        page_view='home',
        initial_query='',
        meta_description=HOME_META_DESCRIPTION
    )

@app.route('/privacy')
@limiter.limit("50 per minute")
def privacy_page():
    """Render the privacy policy page"""
    return render_template('privacy.html')

@app.route('/watchlist')
@limiter.limit("50 per minute")
def watchlist_page():
    """Render the watchlist page"""
    return render_template('index.html', page_view='watchlist', initial_query='')


@app.route('/results')
@limiter.limit("50 per minute")
def search_results_page():
    """Render the dedicated search results page"""
    query = (request.args.get('q') or '').strip()
    return render_template('index.html', page_view='search', initial_query=query)


@app.route('/trending-list')
@limiter.limit("50 per minute")
def trending_board_page():
    """Render the dedicated trending dashboards page"""
    return render_template(
        'index.html',
        page_view='trending',
        initial_query='',
        trending_source='stocktwits',
        meta_description=TRENDING_META_DESCRIPTION
    )


@app.route('/trending-list/<string:source>')
@limiter.limit("50 per minute")
def trending_board_page_source(source):
    """Render the trending dashboards page with a specific source active."""
    source_text = (source or '').strip()
    normalized = _normalize_trending_page_source(source_text)
    if source_text != normalized:
        return redirect(url_for('trending_board_page_source', source=normalized))
    return render_template(
        'index.html',
        page_view='trending',
        initial_query='',
        trending_source=normalized,
        meta_description=TRENDING_META_DESCRIPTION
    )


@app.route('/market-summary')
@limiter.limit("50 per minute")
def market_summary_page():
    """Render the market summary landing page"""
    return render_template(
        'index.html',
        page_view='market',
        initial_query='',
        market_summary_slug=None,
        initial_market_summary_article=None,
        meta_description=MARKET_META_DESCRIPTION
    )


@app.route('/market-summary/stock-market-today')
@limiter.limit("50 per minute")
def market_summary_stock_market_today_page():
    """Render the Stock Market Today landing page with the latest summary."""
    latest_record = get_latest_market_summary_record()
    canonical_url = url_for('market_summary_stock_market_today_page', _external=True)
    page_title = "Stock Market Today"
    if not latest_record:
        return render_template(
            'index.html',
            page_view='market',
            initial_query='',
            market_summary_slug=None,
            initial_market_summary_article=None,
            market_summary_page_title=page_title,
            market_summary_canonical_url=canonical_url,
            meta_description=STOCK_MARKET_TODAY_META_DESCRIPTION
        )
    serialized = serialize_market_summary(latest_record)
    return render_template(
        'index.html',
        page_view='market',
        initial_query='',
        market_summary_slug=serialized.get('slug'),
        initial_market_summary_article=serialized,
        market_summary_page_title=page_title,
        market_summary_canonical_url=canonical_url,
        meta_description=STOCK_MARKET_TODAY_META_DESCRIPTION
    )


@app.route('/market-summary/<string:summary_slug>')
@limiter.limit("50 per minute")
def market_summary_article_page(summary_slug):
    """Render a dedicated market summary article page."""
    record = resolve_market_summary_by_slug(summary_slug)
    if not record:
        return render_template('404.html'), 404
    serialized = serialize_market_summary(record)
    canonical_slug = serialized.get('slug') or summary_slug
    page_title = serialized.get('title') or "Market Summary"
    return render_template(
        'index.html',
        page_view='market',
        initial_query='',
        market_summary_slug=canonical_slug,
        initial_market_summary_article=serialized,
        market_summary_page_title=page_title,
        meta_description=MARKET_META_DESCRIPTION
    )


@app.route('/blog', methods=['GET'])
@limiter.limit("50 per minute")
def blog_listing_page():
    articles = _fetch_published_blog_articles()
    return render_template(
        'blog.html',
        view_mode='list',
        articles=articles,
        meta_description=BLOG_LIST_META_DESCRIPTION,
        page_title='Market Notes Blog'
    )


@app.route('/blog/<string:slug>', methods=['GET'])
@limiter.limit("50 per minute")
def blog_article_page(slug):
    article = _fetch_blog_article_by_slug(slug)
    if not article:
        return render_template('404.html'), 404
    recent_articles = _fetch_published_blog_articles(limit=6)
    related_articles = [item for item in recent_articles if item.slug != article.slug][:4]
    return render_template(
        'blog.html',
        view_mode='detail',
        articles=recent_articles,
        related_articles=related_articles,
        article=article,
        meta_description=article.title,
        page_title=article.title
    )


@app.route('/write', methods=['GET'])
@limiter.limit("20 per minute")
def writer_portal():
    """Render the internal writing surface."""
    return render_template(
        'write.html',
        is_authenticated=_is_blog_admin(),
        default_author=DEFAULT_ARTICLE_AUTHOR,
        meta_description=BLOG_PAGE_META_DESCRIPTION
    )


@app.route('/write/login', methods=['POST'])
@limiter.limit("10 per hour")
def writer_login():
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled. Configure BLOG_ADMIN credentials."}), 503

    payload = request.get_json(silent=True) or request.form or {}
    username = (payload.get('username') or '').strip()
    password = (payload.get('password') or '').strip()
    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if _verify_blog_credentials(username, password):
        session['blog_admin_authenticated'] = True
        session.permanent = True
        return jsonify({"message": "Authenticated."})

    return jsonify({"error": "Invalid credentials."}), 401


@app.route('/write/logout', methods=['POST'])
@limiter.limit("20 per hour")
def writer_logout():
    session.pop('blog_admin_authenticated', None)
    return jsonify({"message": "Logged out."})


@app.route('/api/blog/articles', methods=['GET', 'POST'])
@limiter.limit("40 per hour")
def blog_articles_endpoint():
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled."}), 503
    if not _is_blog_admin():
        return jsonify({"error": "Authentication required."}), 401

    if request.method == 'GET':
        try:
            raw_limit = request.args.get('limit')
            limit = int(raw_limit) if raw_limit is not None else BLOG_ARTICLE_FETCH_LIMIT
        except ValueError:
            limit = BLOG_ARTICLE_FETCH_LIMIT
        limit = max(1, min(limit, BLOG_ARTICLE_FETCH_LIMIT))
        articles = _fetch_recent_blog_articles(limit)
        return jsonify({
            "count": len(articles),
            "articles": [article.to_dict() for article in articles]
        })

    payload = request.get_json(silent=True) or {}
    title = (payload.get('title') or '').strip()
    content = (payload.get('content') or '').strip()
    image_url = (payload.get('image_url') or '').strip()
    author = (payload.get('author') or '').strip() or DEFAULT_ARTICLE_AUTHOR

    if not title:
        return jsonify({"error": "Title is required."}), 400
    if not content:
        return jsonify({"error": "Content is required."}), 400

    try:
        article = _create_blog_article(title, content, image_url, author)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app_logger.error("Unable to save blog article: %s", exc)
        return jsonify({"error": "Unable to save article right now."}), 500

    return jsonify({"article": article.to_dict()}), 201


@app.route('/api/blog/articles/<article_identifier>', methods=['DELETE'])
@limiter.limit("40 per hour")
def blog_article_detail_endpoint(article_identifier):
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled."}), 503
    if not _is_blog_admin():
        return jsonify({"error": "Authentication required."}), 401

    try:
        deleted = _delete_blog_article(article_identifier)
    except Exception:
        return jsonify({"error": "Unable to delete draft right now."}), 500

    if not deleted:
        return jsonify({"error": "Draft not found."}), 404

    return jsonify({"message": "Draft deleted."}), 200


@app.route('/api/blog/articles/<article_identifier>', methods=['PUT', 'PATCH'])
@limiter.limit("40 per hour")
def blog_article_update_endpoint(article_identifier):
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled."}), 503
    if not _is_blog_admin():
        return jsonify({"error": "Authentication required."}), 401

    payload = request.get_json(silent=True) or {}
    title = (payload.get('title') or '').strip()
    content = (payload.get('content') or '').strip()
    image_url = (payload.get('image_url') or '').strip()
    author = (payload.get('author') or '').strip() or DEFAULT_ARTICLE_AUTHOR

    if not title:
        return jsonify({"error": "Title is required."}), 400
    if not content:
        return jsonify({"error": "Content is required."}), 400

    article = _resolve_blog_article(article_identifier)
    if not article:
        return jsonify({"error": "Draft not found."}), 404

    article.title = title
    article.content = content
    article.image_url = image_url or None
    article.author = author
    article.updated_at = _utcnow_naive()

    try:
        db.session.add(article)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app_logger.error("Unable to update blog article %s: %s", article_identifier, exc)
        return jsonify({"error": "Unable to save updates right now."}), 500

    return jsonify({"article": article.to_dict()}), 200


@app.route('/api/blog/articles/<article_identifier>/publish', methods=['POST'])
@limiter.limit("40 per hour")
def blog_article_publish_endpoint(article_identifier):
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled."}), 503
    if not _is_blog_admin():
        return jsonify({"error": "Authentication required."}), 401

    try:
        article = _publish_blog_article(article_identifier)
    except Exception:
        return jsonify({"error": "Unable to publish right now."}), 500

    if not article:
        return jsonify({"error": "Draft not found."}), 404

    return jsonify({
        "article": article.to_dict(),
        "message": "Article posted to the blog."
    })


@app.route('/api/blog/articles/<article_identifier>/unpublish', methods=['POST'])
@limiter.limit("40 per hour")
def blog_article_unpublish_endpoint(article_identifier):
    if not _blog_credentials_configured():
        return jsonify({"error": "Writer access is disabled."}), 503
    if not _is_blog_admin():
        return jsonify({"error": "Authentication required."}), 401

    try:
        article = _unpublish_blog_article(article_identifier)
    except Exception:
        return jsonify({"error": "Unable to unpublish right now."}), 500

    if not article:
        return jsonify({"error": "Draft not found."}), 404

    return jsonify({
        "article": article.to_dict(),
        "message": "Article reverted to draft."
    }), 200


@app.route('/api/market-summary/subscribe', methods=['POST'])
@limiter.limit("3 per hour")
def market_summary_subscribe():
    if not MAILGUN_ENABLED:
        return jsonify({"error": "Email delivery is unavailable right now. Please try again later."}), 503

    payload = request.get_json(silent=True) or {}
    email = (payload.get('email') or '').strip().lower()
    if not _is_valid_email_address(email):
        return jsonify({"error": "Enter a valid email address."}), 400

    try:
        add_member_to_mailgun_list(email)
    except Exception as exc:
        app_logger.error("Mailgun list add failed for %s: %s", email, exc)
        return jsonify({"error": "We couldn't add your email to the list. Please try again shortly."}), 502

    summary_sent = False
    try:
        latest_summary = get_latest_market_summary_record()
        if latest_summary:
            send_market_summary_to_recipient(latest_summary, email)
            summary_sent = True
        else:
            app_logger.info("No market summary available to send for new subscriber %s", email)
    except Exception as exc:
        app_logger.error("Failed to send welcome market summary to %s: %s", email, exc)

    if summary_sent:
        message = (
            "You're subscribed! The latest market summary is on its way, check you spam if you don't see it."
        )
    else:
        message = (
            "You're subscribed! We'll email weekday market summaries after 4:15 PM ET. Make sure to check your spam folder."
        )

    return jsonify({
        "message": message
    })


@app.route('/confirm', methods=['GET'])
@limiter.limit("200 per hour")
def confirm_market_summary_subscription():
    message = (
        "Email confirmations are no longer required. If you entered your email on the Market Summary page, "
        "you're already on the list for weekday recaps after 4:15 PM ET."
    )
    return render_template('confirm.html', status='success', message=message)


@app.route('/search', methods=['POST'])
@limiter.limit("10 per minute")
def search_stock():
    '''Search for stock and return info, historical data, news, and sentiment analysis'''
    request_start = time.time()
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            app_logger.warning("/search payload is not valid JSON")
            return jsonify({'error': 'Request body must be valid JSON'}), 400

        company_name = (payload.get('company_name') or '').strip()
        if not company_name:
            app_logger.warning("/search missing company_name field")
            return jsonify({'error': 'Company name is required'}), 400

        app_logger.info("/search requested for %s", company_name)

        # Search for stock symbol
        results = search(company_name)
        if not results.get('quotes') or len(results['quotes']) == 0:
            app_logger.info("/search no ticker match for %s", company_name)
            return jsonify({'error': 'Company not found'}), 404

        stock_symbol = results['quotes'][0]['symbol'].upper()

        # Get stock information
        stock_info = get_stock_info(stock_symbol)
        if not stock_info:
            app_logger.error("/search unable to hydrate stock info for %s", stock_symbol)
            return jsonify({'error': 'Failed to retrieve stock information'}), 500

        # Get historical data for 1 day
        historical_data = get_historical_data(stock_symbol, '1d')

        response_data = {
            'stock_info': stock_info,
            'historical_data': historical_data,
            'articles': get_news_articles(stock_symbol, 10)
        }

        movement_insight = build_movement_insight(stock_info)
        if movement_insight:
            response_data['movement_insight'] = movement_insight
        app_logger.info(
            "/search completed for %s (%s) in %.2fs",
            company_name,
            stock_symbol,
            time.time() - request_start
        )
        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        app_logger.exception("/search failed for %s", company_name if 'company_name' in locals() else 'unknown')
        return jsonify({'error': str(e)}), 500


@app.route('/historical/<symbol>/<period>', methods=['GET'])
@limiter.limit("50 per minute")
def get_historical(symbol, period):
    """Get historical data for a specific period"""
    try:
        data = get_historical_data(symbol, period)
        return jsonify({'data': data})
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/sentiment', methods=['POST'])
@limiter.limit("30 per minute")
def analyze_article_sentiment():
    """Return sentiment for a single article so the UI can stream updates."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    title = (payload.get('title') or '').strip()
    description = payload.get('description') or ''
    company_name = (payload.get('company_name') or payload.get('symbol') or '').strip()
    article_id = payload.get('article_id')
    raw_run_id = payload.get('sentiment_run_id')
    if raw_run_id is None:
        sentiment_run_id = None
    else:
        sentiment_run_id = str(raw_run_id).strip() or None

    if not title:
        return jsonify({'error': 'Title is required for sentiment analysis'}), 400
    if not company_name:
        company_name = 'the company'

    try:
        sentiment = analyze_sentiment_gemma(title, description, company_name, run_id=sentiment_run_id)
    except RuntimeError as sentiment_exc:
        if str(sentiment_exc) == 'MODEL_OVERLOADED':
            return jsonify({'error': 'The AI model is overloaded. Please try again later.'}), 503
        raise

    response = {'sentiment': sentiment}
    if article_id is not None:
        response['article_id'] = article_id
    return jsonify(response)


# Trending stocks API endpoint
@app.route('/trending', methods=['GET'])
@limiter.limit("30 per minute")
def trending_stocks():
    include_prices = _should_include_prices(request.args.get('include_prices'))
    return jsonify({
        "stocktwits": get_trending_source_data("stocktwits", include_prices=include_prices) or [],
        "reddit": get_trending_source_data("reddit", include_prices=include_prices) or [],
        "volume": get_trending_source_data("volume", include_prices=include_prices) or []
    })


@app.route('/api/market-summary/latest', methods=['GET'])
@limiter.limit("30 per minute")
def market_summary_latest():
    latest = MarketSummary.query.order_by(MarketSummary.summary_date.desc()).first()
    if not latest:
        return jsonify({
            "error": "Market summary unavailable yet. New articles publish at 4:15 PM ET on trading days."
        }), 404
    return jsonify({"article": serialize_market_summary(latest)})


@app.route('/api/market-summary/archive', methods=['GET'])
@limiter.limit("30 per minute")
def market_summary_archive():
    try:
        limit = int(request.args.get('limit', 20))
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 60))
    records = (
        MarketSummary.query.order_by(MarketSummary.summary_date.desc())
        .limit(limit)
        .all()
    )
    return jsonify({
        "count": len(records),
        "articles": [serialize_market_summary(rec) for rec in records]
    })


@app.route('/api/market-summary/<string:summary_slug>', methods=['GET'])
@limiter.limit("30 per minute")
def market_summary_article(summary_slug):
    record = resolve_market_summary_by_slug(summary_slug)
    if not record:
        return jsonify({
            "error": "Market summary article not found."
        }), 404
    return jsonify({"article": serialize_market_summary(record)})


@app.route('/quote/<symbol>', methods=['GET'])
@limiter.limit("300 per minute")
def quote(symbol):
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return jsonify({"error": "Symbol is required"}), 400

    snapshot = get_price_change_snapshot(normalized)
    price = snapshot.get("price")
    if price is None:
        return jsonify({"error": "Price unavailable"}), 404

    return jsonify({
        "symbol": normalized,
        "price": price,
        "change_percent": snapshot.get("change_percent"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route('/trending/<source>', methods=['GET'])
@limiter.limit("30 per minute")
def trending_stocks_source(source):
    include_prices = _should_include_prices(request.args.get('include_prices'))
    data = get_trending_source_data(source, include_prices=include_prices)
    if data is None:
        return jsonify({"error": "Unknown trending source"}), 404
    return jsonify({
        "source": (source or "").lower(),
        "data": data
    })


@app.route('/stocktwits/<symbol>/summary', methods=['GET'])
@limiter.limit("30 per minute")
def stocktwits_summary(symbol):
    normalized = (symbol or "").upper()
    entry = get_stocktwits_summary_entry(normalized)
    if not entry:
        summary_logger.warning(
            "[StockTwits] Summary endpoint miss for %s (no cache entry)",
            normalized
        )
        return jsonify({"error": "Summary unavailable"}), 404
    summary_text = (entry.get("summary") or "").strip()
    used_fallback = False
    if not summary_text:
        summary_text = "No StockTwits summary available yet."
        used_fallback = True
    summary_logger.info(
        "[StockTwits] Served summary for %s (fallback=%s)",
        normalized,
        used_fallback
    )
    return jsonify({
        "ticker": entry.get("ticker"),
        "name": entry.get("name"),
        "summary": summary_text,
        "summary_meta": entry.get("summary_meta"),
        "watchlist_count": entry.get("watchlist_count")
    })

@app.route('/robots.txt', methods=['GET'])
@limiter.limit("100 per minute")
def robots_txt():
    '''Serve robots.txt file'''
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "robots.txt")

@app.route('/sitemap.xml', methods=['GET'])
@limiter.limit("100 per minute")
def sitemap_xml():
    '''Serve sitemap.xml file'''
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "sitemap.xml")


if __name__ == '__main__':
    ensure_market_runtime_bootstrap()
    app.run(
        host='0.0.0.0', 
        debug=False, 
    )