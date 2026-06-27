from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sellsmart_ml.storage.client import get_supabase


def normalize_symbol(value: str) -> str:
    return "".join(ch for ch in value.upper().strip() if ch.isalnum())


def cache_symbol_results(items: list[dict[str, Any]]) -> None:
    """Persist symbol search results for faster autocomplete later."""
    if not items:
        return

    supabase = get_supabase()
    now = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []

    for item in items:
        provider = item.get("provider")
        provider_symbol = item.get("provider_symbol")
        symbol = item.get("symbol")
        name = item.get("name")

        if not provider or not provider_symbol or not symbol or not name:
            continue

        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": item.get("exchange"),
                "exchange_name": item.get("exchange_name"),
                "country": item.get("country"),
                "currency": item.get("currency"),
                "instrument_type": item.get("instrument_type"),
                "logo_url": item.get("logo_url"),
                "provider": provider,
                "provider_symbol": provider_symbol,
                "normalized_symbol": normalize_symbol(symbol),
                "last_loaded_at": now,
            }
        )

    if not rows:
        return

    supabase.table("symbol_cache").upsert(
        rows,
        on_conflict="provider,provider_symbol",
    ).execute()


def search_cached_symbols(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search previously cached symbols by ticker or company name."""
    q = query.strip()
    if not q:
        return []

    # Keep the PostgREST OR filter safe for autocomplete input.
    q_filter = q.replace(",", " ").replace("%", "").replace("*", "").strip()
    if not q_filter:
        return []

    supabase = get_supabase()
    normalized = normalize_symbol(q_filter)

    # PostgREST OR syntax. Search exact/prefix symbol and fuzzy company name.
    response = (
        supabase.table("symbol_cache")
        .select(
            "symbol,name,exchange,exchange_name,country,currency,"
            "instrument_type,logo_url,provider,provider_symbol,last_loaded_at"
        )
        .or_(
            f"normalized_symbol.ilike.{normalized}%,"
            f"symbol.ilike.{q_filter}%,"
            f"name.ilike.%{q_filter}%"
        )
        .limit(limit)
        .execute()
    )

    return response.data or []
