from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sellsmart_ml.storage.client import get_supabase


def get_company_news_from_supabase(
    ticker: str,
    days_back: int = 7,
    limit: int = 50,
) -> pd.DataFrame:
    supabase = get_supabase()

    ticker = ticker.upper()
    from_date = date.today() - timedelta(days=days_back)

    response = (
        supabase
        .table("company_news")
        .select("*")
        .eq("ticker", ticker)
        .gte("news_date", from_date.isoformat())
        .order("news_date", desc=True)
        .limit(limit)
        .execute()
    )

    rows = response.data or []

    if not rows:
        return pd.DataFrame()

    records = []

    for row in rows:
        headline = row.get("headline") or ""
        summary = row.get("summary") or ""

        text = f"{headline}. {summary}".strip(". ").strip()

        if not text:
            continue

        records.append(
            {
                "ticker": ticker,
                "date": pd.to_datetime(row.get("news_date")).normalize(),
                "published_at": row.get("published_at") or row.get("news_date"),
                "headline": headline,
                "summary": summary,
                "text": text,
                "source": row.get("source"),
                "url": row.get("url"),
                "image_url": row.get("image_url") or row.get("image") or row.get("thumbnail_url"),
                "news_status": "supabase",
            }
        )

    return pd.DataFrame(records)