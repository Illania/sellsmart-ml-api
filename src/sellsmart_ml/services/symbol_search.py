from __future__ import annotations

import os
from typing import Any

import requests

from sellsmart_ml.storage.supabase_symbols import cache_symbol_results, search_cached_symbols

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _logo_url(symbol: str) -> str | None:
    # FMP exposes image URLs by ticker. It may not work for every global symbol,
    # so the frontend should keep an initials fallback.
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
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
        "logo_url": _logo_url(symbol),
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
        "logo_url": _logo_url(symbol),
        "provider": "fmp",
        "provider_symbol": symbol.upper(),
    }


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []

    for item in items:
        key = (item["provider"], item["provider_symbol"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


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
    return [item for item in mapped if item is not None][:limit]


def _search_fmp(query: str, limit: int) -> list[dict[str, Any]]:
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    response = requests.get(
        f"{FMP_BASE_URL}/search-symbol",
        params={"query": query, "limit": limit, "apikey": api_key},
        timeout=8,
    )
    response.raise_for_status()

    raw_items = response.json() or []
    mapped = [_map_fmp_item(item) for item in raw_items]
    return [item for item in mapped if item is not None][:limit]


def search_symbols(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search global stocks/ETFs by ticker or company name.

    Returns cached results first when available, then enriches from configured
    providers and stores provider results in Supabase.
    """
    q = query.strip()
    if len(q) < 1:
        return []

    limit = max(1, min(limit, 25))

    cached = search_cached_symbols(q, limit=limit)

    provider_items: list[dict[str, Any]] = []
    provider_errors: list[str] = []

    for provider_search in (_search_twelve_data, _search_fmp):
        try:
            provider_items.extend(provider_search(q, limit=limit))
        except Exception as exc:
            provider_errors.append(str(exc))

    provider_items = _dedupe(provider_items)

    try:
        cache_symbol_results(provider_items)
    except Exception as exc:
        provider_errors.append(f"cache write failed: {exc}")

    combined = _dedupe([*provider_items, *cached])[:limit]

    # Avoid leaking provider exception details to the UI; log for Render.
    if provider_errors:
        print(f"[symbol_search] provider/cache errors for '{q}': {provider_errors}")

    return combined
