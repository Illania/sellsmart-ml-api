from __future__ import annotations

import os
from typing import Iterable

from sellsmart_ml.storage.client import get_supabase

DEFAULT_REFRESH_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
    "CRM",
    "ADBE",
    "INTC",
    "QCOM",
    "PYPL",
    "INSM",
]

MAX_BACKGROUND_TICKERS = int(os.getenv("MAX_BACKGROUND_TICKERS", "100"))


def normalize_ticker(value: object) -> str | None:
    ticker = str(value or "").upper().strip()
    return ticker or None


def unique_tickers(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []

    for value in values:
        ticker = normalize_ticker(value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        results.append(ticker)

    return results


def _select_column(table: str, column: str) -> list[str]:
    try:
        response = get_supabase().table(table).select(column).execute()
        return [row.get(column) for row in (response.data or []) if row.get(column)]
    except Exception as exc:
        print(f"[ticker-universe] Could not load {table}.{column}: {exc}")
        return []


def get_background_refresh_tickers() -> list[str]:
    """Build a daily refresh universe from app data, with MVP defaults first.

    Sources:
    - default MVP tickers, so cron still works on a fresh database
    - latest_predictions, so user-added cached tickers stay fresh
    - positions and watchlist, so active user holdings are refreshed
    - tickers/symbol_cache if present, mainly for logo/search cache continuity
    """
    candidates: list[object] = []
    candidates.extend(DEFAULT_REFRESH_TICKERS)
    candidates.extend(_select_column("latest_predictions", "ticker"))
    candidates.extend(_select_column("positions", "ticker"))
    candidates.extend(_select_column("watchlist", "ticker"))
    candidates.extend(_select_column("tickers", "ticker"))
    candidates.extend(_select_column("symbol_cache", "symbol"))

    return unique_tickers(candidates)[:MAX_BACKGROUND_TICKERS]
