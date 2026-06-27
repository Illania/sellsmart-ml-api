from __future__ import annotations

import os
import re
from typing import Any

import requests

from sellsmart_ml.storage.supabase_symbols import cache_symbol_results, search_cached_symbols

TWELVE_DATA_BASE_URL = "https://api.twelvedata.com"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"

PRIMARY_EXCHANGES = {
    "NASDAQ": 100,
    "NYSE": 95,
    "AMEX": 80,
    "NYSEARCA": 78,
    "LSE": 75,
    "TSX": 72,
    "TSE": 70,
    "JPX": 70,
    "XETR": 68,
    "XETRA": 68,
    "FWB": 65,
    "HKEX": 65,
    "HKSE": 65,
    "ASX": 63,
    "SSE": 60,
    "SZSE": 60,
    "SGX": 58,
    "EURONEXT": 56,
}

# These exchanges are valid, but they are usually secondary cross-listings for
# large US names. Keep them searchable, but do not show them before the primary
# listing for broad company-name searches like "apple".
SECONDARY_LISTING_EXCHANGES = {
    "BMV", "BVC", "TSX", "LSE", "XETR", "XETRA", "FWB", "BER", "MUN", "STU", "DUS",
}

FUNDISH_TYPES = {
    "fund",
    "mutual fund",
    "unit trust",
    "closed-end fund",
}

GOOD_TYPES = {
    "stock",
    "common stock",
    "equity",
    "etf",
    "exchange traded fund",
}

EXCHANGE_ALIASES = {
    "XNGS": "NASDAQ",
    "XNAS": "NASDAQ",
    "XNYS": "NYSE",
    "ARCX": "NYSEARCA",
    "XLON": "LSE",
    "XTSE": "TSX",
    "XTKS": "TSE",
    "XJPX": "JPX",
    "XETR": "XETR",
    "XFRA": "FWB",
    "XHKG": "HKEX",
    "XASX": "ASX",
    "XSES": "SGX",
    "XMEX": "BMV",
}


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _norm(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "", (value or "").upper())


def _norm_words(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", (value or "").upper())).strip()


def _clean_company_name(value: str | None) -> str:
    """Normalize company names for grouping duplicate/cross-listed results."""
    words = _norm_words(value)
    suffixes = {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "PLC", "LTD", "LIMITED",
        "SA", "SE", "NV", "AG", "ADR", "ADS", "DR", "DEPOSITARY", "RECEIPT",
    }
    kept = [word for word in words.split() if word not in suffixes]
    return " ".join(kept) or words


def _canonical_exchange(value: str | None) -> str | None:
    raw = _clean(value)
    if not raw:
        return None
    upper = raw.upper()
    return EXCHANGE_ALIASES.get(upper, upper)


def _logo_url(symbol: str, exchange: str | None = None) -> str | None:
    """Return FMP logo URL only for simple US-style tickers.

    FMP image URLs are often wrong for secondary/global listings. The frontend
    already has an initials fallback, so it is better to return no logo than a
    misleading/broken one.
    """
    clean_symbol = symbol.strip().upper()
    if not clean_symbol or not re.fullmatch(r"[A-Z]{1,5}", clean_symbol):
        return None

    if exchange and _canonical_exchange(exchange) not in {"NASDAQ", "NYSE", "AMEX", "NYSEARCA"}:
        return None

    return f"https://images.financialmodelingprep.com/symbol/{clean_symbol}.png"


def _map_twelve_data_item(item: dict[str, Any]) -> dict[str, Any] | None:
    symbol = _clean(item.get("symbol"))
    name = _clean(item.get("instrument_name")) or _clean(item.get("name"))

    if not symbol or not name:
        return None

    exchange = _canonical_exchange(item.get("exchange") or item.get("mic_code"))
    mic_code = _clean(item.get("mic_code"))
    provider_symbol = symbol if not mic_code else f"{symbol}:{mic_code}"

    return {
        "symbol": symbol.upper(),
        "name": name,
        "exchange": exchange,
        "exchange_name": exchange,
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

    exchange = _canonical_exchange(item.get("exchangeShortName") or item.get("exchange"))

    return {
        "symbol": symbol.upper(),
        "name": name,
        "exchange": exchange,
        "exchange_name": _clean(item.get("exchange")) or exchange,
        "country": None,
        "currency": _clean(item.get("currency")),
        "instrument_type": (_clean(item.get("type")) or "stock").lower(),
        "logo_url": _logo_url(symbol, exchange),
        "provider": "fmp",
        "provider_symbol": symbol.upper(),
    }


def _dedupe_provider_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []

    for item in items:
        provider = item.get("provider") or ""
        provider_symbol = item.get("provider_symbol") or item.get("symbol") or ""
        key = (provider, provider_symbol)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)

    return result


def _is_explicit_exchange_query(query: str) -> bool:
    q = query.upper()
    known = set(PRIMARY_EXCHANGES) | SECONDARY_LISTING_EXCHANGES | set(EXCHANGE_ALIASES)
    return any(exchange in q.split() or f".{exchange}" in q or f":{exchange}" in q for exchange in known)


def _looks_like_symbol_format(query: str) -> bool:
    """Return true when the text looks like a typed ticker/exchange code.

    Important: words like "Apple" or "Tesla" are alphabetic and short, but
    they are still company-name searches in the UI. The previous implementation
    treated any 1-10 character word as a ticker, so "Apple" bypassed company
    grouping and showed every cross-listing.
    """
    text = query.strip()
    if not text or " " in text:
        return False

    # Explicit exchange/ticker notation should stay exchange-specific.
    if any(mark in text for mark in (".", ":", "-")):
        return True

    # All-uppercase short inputs are likely tickers: AAPL, MSFT, VUSA.
    # Lower/mixed case words like Apple, apple, Tesla are treated as names
    # unless an exact symbol match is found in the returned provider results.
    return bool(re.fullmatch(r"[A-Z0-9]{1,8}", text))


def _score_item(item: dict[str, Any], query: str) -> int:
    q_norm = _norm(query)
    q_words = _norm_words(query)
    symbol = (item.get("symbol") or "").upper()
    symbol_norm = _norm(symbol)
    name = item.get("name") or ""
    name_norm = _norm(name)
    name_words = _norm_words(name)
    exchange = _canonical_exchange(item.get("exchange")) or ""
    instrument_type = (item.get("instrument_type") or "").lower()

    score = 0

    if symbol_norm == q_norm:
        score += 140
    elif symbol_norm.startswith(q_norm):
        score += 85

    if name_words == q_words:
        score += 130
    elif name_words.startswith(q_words + " ") or name_words.startswith(q_words):
        score += 110
    elif q_norm and q_norm in name_norm:
        score += 55

    score += PRIMARY_EXCHANGES.get(exchange, 0)

    if instrument_type in {"common stock", "stock", "equity"}:
        score += 35
    elif instrument_type in {"etf", "exchange traded fund"}:
        score += 10
    elif instrument_type in FUNDISH_TYPES:
        score -= 90
    elif "fund" in instrument_type:
        score -= 60

    upper_name = name.upper()
    if " FUND" in upper_name or upper_name.endswith("FUND"):
        score -= 80
    if "SHORT" in upper_name or "LEVERAGED" in upper_name or "3X" in upper_name:
        score -= 90

    # Prefer cleaner US primary listings for broad name searches.
    if not _looks_like_symbol_format(query) and exchange in SECONDARY_LISTING_EXCHANGES:
        score -= 35

    return score


def _rank_and_filter(items: list[dict[str, Any]], query: str, limit: int) -> list[dict[str, Any]]:
    """Rank search results for normal retail-investor expectations.

    For broad company-name searches, show one best listing per company by
    default, usually the primary listing. Secondary listings remain discoverable
    when the user searches by ticker/exchange explicitly.
    """
    if not items:
        return []

    q_norm = _norm(query)
    symbol_format_query = _looks_like_symbol_format(query)
    explicit_exchange = _is_explicit_exchange_query(query)

    # Remove exact technical duplicates first, regardless of provider.
    exact_seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        symbol = _norm(item.get("symbol"))
        exchange = _canonical_exchange(item.get("exchange")) or ""
        name_key = _clean_company_name(item.get("name"))
        key = (symbol, exchange, name_key)
        if key in exact_seen:
            continue
        exact_seen.add(key)
        unique.append(item)

    scored = [(item, _score_item(item, query)) for item in unique]

    # Discard weak matches that only appear because provider search is broad.
    # Keep exact ticker matches even when their name does not contain the query.
    strong: list[tuple[dict[str, Any], int]] = []
    for item, score in scored:
        symbol_norm = _norm(item.get("symbol"))
        name_norm = _norm(item.get("name"))
        instrument_type = (item.get("instrument_type") or "").lower()

        direct_match = symbol_norm.startswith(q_norm) or q_norm in name_norm
        exact_ticker = symbol_norm == q_norm

        if not direct_match and not exact_ticker:
            continue

        # For ordinary searches, avoid mutual funds unless the user searched the
        # exact fund ticker/name. This stops "Appleseed Fund" outranking Apple.
        if instrument_type in FUNDISH_TYPES or "fund" in instrument_type:
            if not exact_ticker and not name_norm.startswith(q_norm):
                continue

        if score < 40:
            continue
        strong.append((item, score))

    strong.sort(key=lambda pair: pair[1], reverse=True)

    # Collapse cross-listings for broad company-name searches.
    #
    # The previous bug: "Apple" matched the old ticker heuristic because it is
    # a short alphabetic word, so the code skipped this grouping and returned
    # Apple on NASDAQ, LSE, XETRA, TSX, etc. For a normal company-name search,
    # users should see the primary listing only.
    #
    # Keep exchange-specific variants only when the user explicitly types an
    # exchange/ticker notation, or when the query exactly equals a returned
    # ticker such as AAPL/MSFT/TSLA.
    has_exact_symbol_match = any(_norm(item.get("symbol")) == q_norm for item, _score in strong)
    should_collapse_by_company = not explicit_exchange and not (symbol_format_query and has_exact_symbol_match)

    if should_collapse_by_company:
        best_by_company: dict[str, tuple[dict[str, Any], int]] = {}
        for item, score in strong:
            company_key = _clean_company_name(item.get("name"))
            current = best_by_company.get(company_key)
            if current is None or score > current[1]:
                best_by_company[company_key] = (item, score)
        strong = sorted(best_by_company.values(), key=lambda pair: pair[1], reverse=True)

    # If the user typed a ticker like AAPL, keep only one result per exchange and
    # prefer the best-ranked representation from providers/cache.
    best_by_symbol_exchange: dict[tuple[str, str], tuple[dict[str, Any], int]] = {}
    for item, score in strong:
        key = (_norm(item.get("symbol")), _canonical_exchange(item.get("exchange")) or "")
        current = best_by_symbol_exchange.get(key)
        if current is None or score > current[1]:
            best_by_symbol_exchange[key] = (item, score)

    ranked = sorted(best_by_symbol_exchange.values(), key=lambda pair: pair[1], reverse=True)
    return [item for item, _score in ranked[:limit]]


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
    return [item for item in mapped if item is not None][: max(limit, 20)]


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
    return [item for item in mapped if item is not None][: max(limit, 20)]


def search_symbols(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search global stocks/ETFs by ticker or company name.

    Returns ranked, normalized suggestions. For broad company-name searches,
    results are collapsed toward the most likely primary listing. For exact
    ticker/exchange searches, secondary listings remain visible.
    """
    q = query.strip()
    if len(q) < 1:
        return []

    limit = max(1, min(limit, 25))

    # Pull a few more than requested so ranking can choose the best candidates.
    provider_limit = max(limit * 3, 20)
    cached = search_cached_symbols(q, limit=provider_limit)

    provider_items: list[dict[str, Any]] = []
    provider_errors: list[str] = []

    for provider_search in (_search_twelve_data, _search_fmp):
        try:
            provider_items.extend(provider_search(q, limit=provider_limit))
        except Exception as exc:
            provider_errors.append(str(exc))

    provider_items = _dedupe_provider_items(provider_items)

    try:
        cache_symbol_results(provider_items)
    except Exception as exc:
        provider_errors.append(f"cache write failed: {exc}")

    combined = _dedupe_provider_items([*provider_items, *cached])
    ranked = _rank_and_filter(combined, q, limit=limit)

    # Avoid leaking provider exception details to the UI; log for Render.
    if provider_errors:
        print(f"[symbol_search] provider/cache errors for '{q}': {provider_errors}")

    return ranked
