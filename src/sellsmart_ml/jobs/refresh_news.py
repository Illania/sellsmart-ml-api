from __future__ import annotations

import hashlib
import os
from datetime import date, timedelta, datetime, timezone

import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "NFLX", "JPM", "CRM", "ADBE", "INTC", "QCOM", "PYPL", "INSM",
]

DAYS_BACK = 30


def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing")

    return create_client(url, key)


def make_news_id(ticker: str, item: dict) -> str:
    raw = f"{ticker}|{item.get('datetime')}|{item.get('headline')}|{item.get('url')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fetch_news(ticker: str) -> list[dict]:
    api_key = os.getenv("FINNHUB_API_KEY")

    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is missing")

    today = date.today()
    from_date = today - timedelta(days=DAYS_BACK)

    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": today.isoformat(),
            "token": api_key,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Finnhub response for {ticker}: {data}")

    return data


def save_news_to_supabase(ticker: str, items: list[dict]) -> None:
    supabase = get_supabase()

    rows = []

    for item in items:
        ts = item.get("datetime")

        news_date = None
        if ts:
            news_date = datetime.fromtimestamp(
                int(ts),
                tz=timezone.utc,
            ).date().isoformat()

        rows.append(
            {
                "id": make_news_id(ticker, item),
                "ticker": ticker,
                "news_date": news_date,
                "datetime_ts": ts,
                "headline": item.get("headline"),
                "summary": item.get("summary"),
                "source": item.get("source"),
                "url": item.get("url"),
                "raw_json": item,
            }
        )

    if not rows:
        print(f"No news rows for {ticker}")
        return

    supabase.table("company_news").upsert(rows).execute()
    print(f"Saved {len(rows)} news rows for {ticker}")


def main() -> None:
    print("Starting news refresh...")

    for ticker in TICKERS:
        print("")
        print(f"Refreshing {ticker}...")

        try:
            items = fetch_news(ticker)
            save_news_to_supabase(ticker, items)

        except Exception as exc:
            print(f"ERROR {ticker}: {exc}")

    print("")
    print("News refresh completed.")


if __name__ == "__main__":
    main()