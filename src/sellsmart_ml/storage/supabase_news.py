from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from sellsmart_ml.storage.client import get_supabase
from sellsmart_ml.storage.supabase_predictions import clean_json_value


SENTIMENT_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "neg_prob",
    "is_negative",
    "is_very_negative",
    "sentiment_model",
    "sentiment_scored_at",
]


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
        .select(
            "id, ticker, news_date, headline, summary, source, url, "
            "sentiment_label, sentiment_score, neg_prob, is_negative, "
            "is_very_negative, sentiment_model, sentiment_scored_at"
        )
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

        record = {
            "news_id": row.get("id"),
            "ticker": ticker,
            "date": pd.to_datetime(row.get("news_date")).normalize(),
            "text": text,
            "source": row.get("source"),
            "url": row.get("url"),
            "news_status": "supabase",
        }

        for col in SENTIMENT_COLUMNS:
            record[col] = row.get(col)

        records.append(record)

    return pd.DataFrame(records)


def update_company_news_sentiment(
    news_id: str,
    *,
    sentiment_label: str,
    sentiment_score: float,
    neg_prob: float,
    is_negative: int | bool,
    is_very_negative: int | bool,
    sentiment_model: str,
) -> None:
    """Persist FinBERT sentiment for one company_news row.

    This is intentionally non-fatal. Prediction generation should continue even
    if Supabase sentiment persistence fails because of a migration/deploy issue.
    """
    if not news_id:
        return

    payload: dict[str, Any] = clean_json_value(
        {
            "sentiment_label": sentiment_label,
            "sentiment_score": float(sentiment_score),
            "neg_prob": float(neg_prob),
            "is_negative": bool(is_negative),
            "is_very_negative": bool(is_very_negative),
            "sentiment_model": sentiment_model,
            "sentiment_scored_at": pd.Timestamp.utcnow().isoformat(),
        }
    )

    try:
        (
            get_supabase()
            .table("company_news")
            .update(payload)
            .eq("id", news_id)
            .execute()
        )
    except Exception as exc:
        print(f"WARNING: failed to persist news sentiment for {news_id}: {exc}")


def update_company_news_sentiments(news_df: pd.DataFrame, sentiment_model: str) -> None:
    if news_df.empty or "news_id" not in news_df.columns:
        return

    persisted = 0

    for _, row in news_df.iterrows():
        news_id = row.get("news_id")
        if not news_id:
            continue

        update_company_news_sentiment(
            str(news_id),
            sentiment_label=str(row.get("sentiment_label") or "neutral"),
            sentiment_score=float(row.get("sentiment_score") or 0.0),
            neg_prob=float(row.get("neg_prob") or 0.0),
            is_negative=int(row.get("is_negative") or 0),
            is_very_negative=int(row.get("is_very_negative") or 0),
            sentiment_model=sentiment_model,
        )
        persisted += 1

    if persisted:
        print(f"Persisted sentiment for {persisted} news row(s).")
