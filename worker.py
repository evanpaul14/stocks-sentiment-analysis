import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerConfig:
    refresh_interval_seconds: int
    health_check_seconds: int
    stale_entry_ttl_seconds: int = 60 * 60


@dataclass(frozen=True)
class WorkerDependencies:
    init_db: Callable[[], None]
    refresh_tracked_trending_symbols: Callable[[], None]
    get_tracked_symbols: Callable[[], Set[str]]
    get_cached_symbols: Callable[[], Set[str]]
    get_stock_info: Callable[[str], Optional[dict]]
    build_sentiment_payload: Callable[..., Tuple[List[dict], dict, str]]
    sentiment_analyzer: Callable[[str, str, str], str]
    save_sentiment_to_db: Callable[[str, List[dict], dict, str], None]
    purge_stale_sentiment_records: Callable[[Iterable[str]], None]
    delete_outdated_entries: Callable[[int], int]


_thread_lock = threading.Lock()
_worker_started = False


def _prune_outdated_entries(deps: WorkerDependencies, ttl_seconds: int) -> None:
    try:
        removed = deps.delete_outdated_entries(ttl_seconds)
        if removed:
            logger.info("Pruned %d outdated sentiment records (>%ds old)", removed, ttl_seconds)
        else:
            logger.debug("No outdated sentiment records found (>%ds old)", ttl_seconds)
    except Exception:
        logger.exception("Failed to prune outdated sentiment cache entries")


def _refresh_sentiment_cache(deps: WorkerDependencies, config: WorkerConfig) -> None:
    _prune_outdated_entries(deps, config.stale_entry_ttl_seconds)
    deps.refresh_tracked_trending_symbols()
    symbols_to_update = deps.get_tracked_symbols() or set()
    if not symbols_to_update:
        logger.warning("No tracked symbols available for cache refresh")
        return

    logger.info("Refreshing sentiment cache for %d tracked symbols", len(symbols_to_update))
    for symbol in symbols_to_update:
        if not symbol:
            continue
        try:
            stock_info = deps.get_stock_info(symbol)
            company_name = stock_info['companyName'] if stock_info else symbol
            articles, sentiment_summary, overall_sentiment = deps.build_sentiment_payload(
                symbol,
                company_name,
                sentiment_analyzer=deps.sentiment_analyzer
            )
            deps.save_sentiment_to_db(symbol, articles, sentiment_summary, overall_sentiment)
            logger.debug("Updated sentiment cache for %s", symbol)
        except Exception:
            logger.exception("Error refreshing sentiment for %s", symbol)

    deps.purge_stale_sentiment_records(symbols_to_update)


def _ensure_cache_completeness(deps: WorkerDependencies, config: WorkerConfig) -> None:
    _prune_outdated_entries(deps, config.stale_entry_ttl_seconds)
    deps.refresh_tracked_trending_symbols()
    tracked_symbols = deps.get_tracked_symbols() or set()
    if not tracked_symbols:
        logger.debug("Skipping cache completeness check because no tracked symbols are available")
        return

    cached_symbols = deps.get_cached_symbols() or set()
    missing_symbols = [symbol for symbol in tracked_symbols if symbol not in cached_symbols]

    if not missing_symbols:
        logger.debug("All %d tracked symbols already cached", len(tracked_symbols))
        return

    logger.info("Ensuring cache completeness for %d missing symbols", len(missing_symbols))
    for symbol in missing_symbols:
        try:
            stock_info = deps.get_stock_info(symbol)
            company_name = stock_info['companyName'] if stock_info else symbol
            articles, sentiment_summary, overall_sentiment = deps.build_sentiment_payload(
                symbol,
                company_name,
                sentiment_analyzer=deps.sentiment_analyzer
            )
            deps.save_sentiment_to_db(symbol, articles, sentiment_summary, overall_sentiment)
            logger.debug("Filled missing sentiment cache for %s", symbol)
        except Exception:
            logger.exception("Error ensuring sentiment cache for %s", symbol)

    deps.purge_stale_sentiment_records(tracked_symbols)


def _worker_loop(deps: WorkerDependencies, config: WorkerConfig) -> None:
    logger.info(
        "Sentiment background worker loop started: refresh=%ds, health_check=%ds",
        config.refresh_interval_seconds,
        config.health_check_seconds,
    )
    last_full_refresh = 0.0
    while True:
        now = time.time()
        if now - last_full_refresh >= config.refresh_interval_seconds:
            logger.info("Starting scheduled full sentiment cache refresh")
            try:
                _refresh_sentiment_cache(deps, config)
            except Exception:
                logger.exception("Background refresh failed")
            else:
                last_full_refresh = now
                logger.info("Completed full sentiment cache refresh")

        try:
            _ensure_cache_completeness(deps, config)
        except Exception:
            logger.exception("Cache completeness check failed")

        logger.debug("Background worker sleeping for %ds", config.health_check_seconds)
        time.sleep(config.health_check_seconds)


def start_background_worker(deps: WorkerDependencies, config: Optional[WorkerConfig] = None) -> None:
    global _worker_started
    if config is None:
        config = WorkerConfig(refresh_interval_seconds=60 * 60, health_check_seconds=10 * 60)

    with _thread_lock:
        if _worker_started:
            logger.info("Background worker already running")
            return

        deps.init_db()
        thread = threading.Thread(
            target=_worker_loop,
            args=(deps, config),
            daemon=True,
            name="sentiment-background-worker",
        )
        thread.start()
        _worker_started = True
        logger.info("Background worker thread started")
