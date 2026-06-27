from __future__ import annotations

import os
import re
from typing import Any

import requests

from sellsmart_ml.storage.supabase_symbols import cache_symbol_results, search_cached_symbols

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

MAJOR_EXCHANGE_SCORE = {
    "NASDAQ": 80,
    "NYSE": 75,
    "AMEX": 65,
    "NYSE ARCA": 60,
    "ARCA": 60,
    "LSE": 45,
    "TSX": 40,
    "TSE": 40,
    "XETR": 35,
    "FWB": 30,
    "HKEX": 30,
    "ASX": 30,
}

US_LOGO_EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "ARCA"}

LEVERAGED_OR_DERIVATIVE_TERMS = (
    " 2X ",
    " 3X ",
    " 4X ",
    "SHORT",
    "INVERSE",
    "LEVERAGED",
    "BULL",
    "BEAR",
    "ULTRA",
    "DAILY",
)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize(value: str | None) -> str:
    return "".join(ch for ch in (value or "").upper().strip() if ch.isalnum())


def _word_contains(text: str, query: str) -> bool:
    return query in text.lower()


def _word_starts(text: str, query: str) -> bool:
    q = re.escape(query.lower())
    return bool(re.search(rf"(^|[^a-z0-9]){q}", text.lower()))


def _logo_url(symbol: str, exchange: str | None = None) -> str | None:
    """Return a logo URL only when it is likely to be correct.

    FMP logo URLs are ticker-based. For foreign listings like Apple on XETR
    with symbol APC, using APC.png can show the wrong company logo. For those
    cases we return None so the frontend can show an initials fallback.
    """
    clean_symbol = symbol.strip().upper()
    clean_exchange = (exchange or "").strip().upper()

    if not clean_symbol or clean_exchange not in US_LOGO_EXCHANGES:
        return None

    return f"https://images.financialmodelingprep.com/symbol/{clean_symbol}.png"


def _map_twelve_data_item(item: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _clean(item.get("symbol"))
    name = _clean(item.get("instrument_name")) or _clean(item.get("name"))

    if not symbol or not name:
        return None

    exchange = _clean(item.get("exchange")) or _clean(item.get("mic_code"))
    mic_code = _clean(item.get("mic_code"))
    provider_symbol = symbol if not mic_code else f"{symbol}:{mic_code}"

    return {
        "symbol": symbol.upper(),
        "name": name,
        "exchange": exchange,
        "exchange_name": _clean(item.get("exchange")),
        "country": _clean(item.get("country")),
        "currency": _clean(item.get("currency")),
        "instrument_type": (_clean(item.get("instrument_type")) or "stock").lower(),
        "logo_url": _logo_url(symbol, exchange),
        "provider": "twelvedata",
        "provider_symbol": provider_symbol.upper(),
    }


def _map_fmp_item(item: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _clean(item.get("symbol"))
    name = _clean(item.get("name"))

    if not symbol or not name:
        return None

    exchange = _clean(item.get("exchangeShortName")) or _clean(item.get("exchange"))

    return {
        "symbol": symbol.upper(),
        "name": name,
        "exchange": exchange,
        "exchange_name": _clean(item.get("exchange")),
        "country": None,
        "currency": _clean(item.get("currency")),
        "instrument_type": (_clean(item.get("type")) or "stock").lower(),
        "logo_url": _logo_url(symbol, exchange),
        "provider": "fmp",
        "provider_symbol": symbol.upper(),
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []

    for item in items:
        key = (item.get("provider") or "", item.get("provider_symbol") or item.get("symbol") or "")
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def _item_score(item: dict[str, Any], query: str) -> int | None:
    """Score a result for autocomplete. Return None to hide weak matches.

    Provider search APIs can return broad thematic ETF results for company-name
    queries. Autocomplete should feel strict: it should show symbols/names that
    actually match what the user typed, then rank primary listings first.
    """
    q_text = query.strip().lower()
    q_norm = _normalize(query)
    symbol = (item.get("symbol") or "").upper()
    symbol_norm = _normalize(symbol)
    name = item.get("name") or ""
    name_lower = name.lower()
    exchange = (item.get("exchange") or "").upper()
    instrument_type = (item.get("instrument_type") or "").lower()

    if not q_text or not q_norm:
        return None

    score: int | None = None

    if symbol_norm == q_norm:
        score = 1000
    elif symbol_norm.startswith(q_norm):
        score = 900
    elif q_norm in symbol_norm and len(q_norm) >= 2:
        score = 720
    elif name_lower == q_text:
        score = 860
    elif name_lower.startswith(q_text):
        score = 840
    elif _word_starts(name, q_text):
        score = 780
    elif _word_contains(name, q_text):
        score = 680

    # Hide broad provider results that do not visibly match symbol or company
    # name. This removes rows like GraniteShares appearing for "apple".
    if score is None:
        return None

    score += MAJOR_EXCHANGE_SCORE.get(exchange, 0)

    if "common stock" in instrument_type or instrument_type in {"stock", "common"}:
        score += 35
    elif "etf" in instrument_type:
        score -= 60
    elif "depositary" in instrument_type or "receipt" in instrument_type:
        score -= 25

    name_padded = f" {name.upper()} "
    if any(term in name_padded for term in LEVERAGED_OR_DERIVATIVE_TERMS):
        score -= 200

    # For company-name searches, keep ETFs only when the ETF name itself
    # visibly matches the query or the user typed the ETF ticker.
    if "etf" in instrument_type and not (symbol_norm.startswith(q_norm) or _word_contains(name, q_text)):
        return None

    return score


def _rank_and_filter(items: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []

    for item in _dedupe(items):
        score = _item_score(item, query)
        if score is None:
            continue
        scored.append((score, item))

    scored.sort(
        key=lambda pair: (
            pair[0],
            pair[1].get("symbol") or "",
            pair[1].get("exchange") or "",
        ),
        reverse=True,
    )

    return [item for _, item in scored[:limit]]


def _search_twelve_data(query: str, limit: int) -> list[dict[str, Any]]:
    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        return []

    response = requests.get(
        f"{TWELVE_DATA_BASE_URL}/symbol_search",
        params={"symbol": query, "apikey": api_key},
        timeout=8,
    )
    response.raise_for_status()

    payload = response.json()
    raw_items = payload.get("data") or []

    mapped = [_map_twelve_data_item(item) for item in raw_items]
    return [item for item in mapped if item is not None]


def _search_fmp(query: str, limit: int) -> list[dict[str, Any]]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    response = requests.get(
        f"{FMP_BASE_URL}/search-symbol",
        params={"query": query, "limit": max(limit, 20), "apikey": api_key},
        timeout=8,
    )
    response.raise_for_status()

    raw_items = response.json() or []
    mapped = [_map_fmp_item(item) for item in raw_items]
    return [item for item in mapped if item is not None]


def search_symbols(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search global stocks/ETFs by ticker or company name."""
    q = query.strip()
    if len(q) < 1:
        return []

    limit = max(1, min(limit, 25))

    cached: list[dict[str, Any]] = []
    provider_items: list[dict[str, Any]] = []
    provider_errors: list[str] = []

    try:
        cached = search_cached_symbols(q, limit=max(limit, 20))
    except Exception as exc:
        provider_errors.append(f"cache read failed: {exc}")

    for provider_search in (_search_twelve_data, _search_fmp):
        try:
            provider_items.extend(provider_search(q, limit=max(limit, 20)))
        except Exception as exc:
            provider_errors.append(str(exc))

    ranked_provider_items = _rank_and_filter(provider_items, q, limit=max(limit, 20))

    try:
        cache_symbol_results(ranked_provider_items)
    except Exception as exc:
        provider_errors.append(f"cache write failed: {exc}")

    combined = _rank_and_filter([*ranked_provider_items, *cached], q, limit=limit)

    # Avoid leaking provider exception details to the UI; log for Render.
    if provider_errors:
        print(f"[symbol_search] provider/cache errors for '{q}': {provider_errors}")

    return combined
