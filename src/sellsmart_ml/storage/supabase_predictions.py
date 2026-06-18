import math
import os
from datetime import datetime, timezone
from typing import Any, Optional

from sellsmart_ml.storage.client import get_supabase


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


def save_latest_prediction(prediction: dict) -> None:
    supabase = get_supabase()

    ticker = prediction["ticker"].upper()
    generated_at = datetime.now(timezone.utc).isoformat()

    prediction = clean_json_value({
        **prediction,
        "generated_at": generated_at,
        "cache_generated_at": generated_at,
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
    prediction = row["prediction_json"]
    prediction["cache_status"] = "supabase"
    prediction["cache_generated_at"] = row["generated_at"]

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
        prediction = row["prediction_json"]
        prediction["cache_status"] = "supabase"
        prediction["cache_generated_at"] = row["generated_at"]

        results.append(prediction)

    return results