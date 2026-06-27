import math
import os
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sellsmart_ml.storage.client import get_supabase


# MVP cache policy:
# Cron predictions are currently refreshed once per day, so 24h cache is a good default.
# Override in Render/Supabase env if needed, e.g. PREDICTION_CACHE_TTL_HOURS=12.
PREDICTION_CACHE_TTL_HOURS = int(os.getenv("PREDICTION_CACHE_TTL_HOURS", "24"))


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {
            str(k): clean_json_value(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            clean_json_value(v)
            for v in value
        ]

    return value


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse Supabase timestamp strings safely."""
    if not value:
        return None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            # Supabase/Postgres may return either "+00:00" or "Z" timestamps.
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def is_prediction_cache_fresh(generated_at: Any) -> bool:
    """Return True only if cached prediction is recent enough for MVP usage."""
    parsed = parse_datetime(generated_at)

    if parsed is None:
        return False

    max_age = timedelta(hours=PREDICTION_CACHE_TTL_HOURS)
    age = datetime.now(timezone.utc) - parsed

    return timedelta(0) <= age <= max_age


def save_latest_prediction(prediction: dict) -> None:
    """Save or update the latest prediction for a ticker.

    This is used both by the daily cron and by live user-requested predictions,
    so user-added tickers become cached after the first slow request.
    """
    supabase = get_supabase()

    ticker = prediction["ticker"].upper()
    generated_at = datetime.now(timezone.utc).isoformat()

    prediction = clean_json_value({
        **prediction,
        "ticker": ticker,
        "generated_at": generated_at,
        "cache_generated_at": generated_at,
        "cache_status": "generated",
    })

    row = clean_json_value({
        "ticker": ticker,
        "generated_at": generated_at,
        "prediction_date": prediction.get("date"),
        "horizon": prediction.get("horizon"),
        "risk_score": prediction.get("risk_score"),
        "probability_of_drop": prediction.get("probability_of_drop"),
        "category": prediction.get("category"),
        "action": prediction.get("action"),
        "confidence": prediction.get("confidence"),
        "news_status": prediction.get("news_status"),
        "market_regime": prediction.get("market_regime"),
        "prediction_json": prediction,
    })

    # Current MVP stores one latest prediction per ticker.
    # If you add multiple horizons later, change DB constraint + on_conflict to "ticker,horizon".
    supabase.table("latest_predictions").upsert(
        row,
        on_conflict="ticker",
    ).execute()


def get_latest_prediction(ticker: str) -> Optional[dict]:
    supabase = get_supabase()

    response = (
        supabase
        .table("latest_predictions")
        .select("*")
        .eq("ticker", ticker.upper())
        .limit(1)
        .execute()
    )

    if not response.data:
        return None

    row = response.data[0]
    generated_at = row.get("generated_at")

    # Important: do not return stale cached predictions.
    # Returning None tells the API endpoint to run a fresh live prediction.
    if not is_prediction_cache_fresh(generated_at):
        return None

    prediction = dict(row["prediction_json"] or {})
    prediction["cache_status"] = "supabase"
    prediction["cache_generated_at"] = generated_at

    return prediction


def get_all_latest_predictions() -> list[dict]:
    supabase = get_supabase()

    response = (
        supabase
        .table("latest_predictions")
        .select("*")
        .order("risk_score", desc=True)
        .execute()
    )

    rows = response.data or []

    results = []

    for row in rows:
        prediction = dict(row["prediction_json"] or {})
        prediction["cache_status"] = "supabase"
        prediction["cache_generated_at"] = row.get("generated_at")
        prediction["cache_is_fresh"] = is_prediction_cache_fresh(row.get("generated_at"))

        results.append(prediction)

    return results
