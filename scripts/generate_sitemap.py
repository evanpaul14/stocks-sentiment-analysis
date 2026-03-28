#!/usr/bin/env python3
"""Generate sitemap.xml from market_summary SQLite data.

This script creates a sitemap that includes core pages plus market-summary pages
based on rows in the market_summary table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = (
    "http://www.sitemaps.org/schemas/sitemap/0.9 "
    "http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd"
)

ET.register_namespace("", SITEMAP_NS)
ET.register_namespace("xsi", XSI_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sitemap.xml from market_summary SQLite table")
    parser.add_argument(
        "--db",
        dest="db_path",
        default="instance/ip_log.db",
        help="Path to SQLite database (default: instance/ip_log.db)",
    )
    parser.add_argument(
        "--output",
        default="sitemap.xml",
        help="Output sitemap path (default: sitemap.xml)",
    )
    parser.add_argument(
        "--base-url",
        default="https://stocksentimentapp.com",
        help="Website base URL (default: https://stocksentimentapp.com)",
    )
    parser.add_argument(
        "--stock-market-today-lastmod",
        default=None,
        help=(
            "Optional YYYY-MM-DD override for /market-summary/stock-market-today <lastmod>. "
            "By default, the latest summary_date from DB is used."
        ),
    )
    return parser.parse_args()


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("base URL cannot be empty")
    return normalized


def _collect_summary_dates(db_path: Path) -> list[str]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    dates: set[str] = set()
    connection = sqlite3.connect(str(db_path))
    try:
        cursor = connection.execute("SELECT summary_date FROM market_summary")
        for row in cursor:
            raw_date = (row[0] or "").strip() if row else ""
            if not raw_date:
                continue
            # Keep only valid ISO date values.
            try:
                parsed = datetime.strptime(raw_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            dates.add(parsed.isoformat())
    finally:
        connection.close()

    return sorted(dates)


def _add_url(
    root: ET.Element,
    loc: str,
    changefreq: str,
    priority: str,
    lastmod: str | None = None,
) -> None:
    url_el = ET.SubElement(root, f"{{{SITEMAP_NS}}}url")

    loc_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}loc")
    loc_el.text = loc

    changefreq_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}changefreq")
    changefreq_el.text = changefreq

    priority_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}priority")
    priority_el.text = priority

    if lastmod:
        lastmod_el = ET.SubElement(url_el, f"{{{SITEMAP_NS}}}lastmod")
        lastmod_el.text = lastmod


def _validate_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
    return parsed.isoformat()


def build_sitemap(
    base_url: str,
    summary_dates: list[str],
    stock_market_today_lastmod: str | None = None,
) -> ET.ElementTree:
    root = ET.Element(
        f"{{{SITEMAP_NS}}}urlset",
        {
            f"{{{XSI_NS}}}schemaLocation": SCHEMA_LOCATION,
        },
    )

    latest_summary_date = stock_market_today_lastmod or (
        summary_dates[-1] if summary_dates else datetime.now(timezone.utc).date().isoformat()
    )

    # Required core URLs.
    _add_url(root, f"{base_url}/", "daily", "1.0")
    _add_url(root, f"{base_url}/trending-list", "daily", "0.9")
    _add_url(root, f"{base_url}/market-summary", "daily", "1.0")
    _add_url(
        root,
        f"{base_url}/market-summary/stock-market-today",
        "daily",
        "1.0",
        lastmod=latest_summary_date,
    )

    # Date-based market summary URLs sourced from the DB.
    for summary_date in summary_dates:
        _add_url(
            root,
            f"{base_url}/market-summary/{summary_date}",
            "daily",
            "0.8",
            lastmod=summary_date,
        )

    return ET.ElementTree(root)


def write_sitemap(tree: ET.ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    base_url = _normalize_base_url(args.base_url)
    stock_market_today_lastmod = _validate_iso_date(args.stock_market_today_lastmod)

    summary_dates = _collect_summary_dates(db_path)
    tree = build_sitemap(base_url, summary_dates, stock_market_today_lastmod)
    write_sitemap(tree, output_path)

    effective_lastmod = stock_market_today_lastmod or (summary_dates[-1] if summary_dates else "(none found)")
    print(f"Generated sitemap: {output_path}")
    print(f"Market summary rows included: {len(summary_dates)}")
    print(f"Date used for stock-market-today lastmod: {effective_lastmod}")


if __name__ == "__main__":
    main()
