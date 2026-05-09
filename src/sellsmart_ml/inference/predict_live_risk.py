from __future__ import annotations

import json
import joblib
import os
import requests
import subprocess
import warnings

from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from pandas.errors import PerformanceWarning
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from sellsmart_ml.config import MODELS_DIR
from sellsmart_ml.dataset.build_price_dataset import (
    add_market_context,
    build_market_context,
)
from sellsmart_ml.features.news_features import add_news_features
from sellsmart_ml.features.price_features import add_price_features
from sellsmart_ml.inference.insight_generator import generate_insight
from sellsmart_ml.storage.supabase_news import get_company_news_from_supabase

warnings.simplefilter("ignore", PerformanceWarning)


# =========================================================
# TIMING
# =========================================================

T0 = perf_counter()


def log_time(step: str):
    print(f"[timing] {step}: {perf_counter() - T0:.2f}s")


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "panic_model_price_plus_news"

MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
FEATURES_PATH = MODELS_DIR / f"{MODEL_NAME}_features.json"
THRESHOLD_PATH = MODELS_DIR / f"{MODEL_NAME}_threshold.json"

FINBERT_MODEL = "ProsusAI/finbert"

PRICE_PERIOD = "1y"
NEWS_HISTORY_DAYS = 7

CACHE_ROOT = Path("data/cache")

NEWS_CACHE_DIR = CACHE_ROOT / "live_news"
PRICE_CACHE_DIR = CACHE_ROOT / "live_prices"
MARKET_CACHE_DIR = CACHE_ROOT / "market_context"

NEWS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD MODEL
# =========================================================

log_time("start")

print("Loading model...")

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)["threshold"]

print("Features:", len(FEATURE_COLUMNS))
print("Threshold:", THRESHOLD)

log_time("model loaded")


# =========================================================
# LOAD FINBERT
# =========================================================

print("Loading FinBERT...")

tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

finbert.to(device)
finbert.eval()

print("Device:", device)

log_time("FinBERT loaded")


# =========================================================
# HELPERS
# =========================================================

def make_fallback_news_row(
    ticker: str,
    status: str = "fallback",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker.upper(),
                "date": pd.Timestamp.today().normalize(),
                "text": f"No recent company news available for {ticker.upper()}.",
                "source": status,
                "url": None,
                "news_status": status,
            }
        ]
    )


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[0] else col[-1]
            for col in df.columns
        ]

    return df


# =========================================================
# DOWNLOAD LIVE PRICE DATA
# =========================================================

def download_live_price_data(
    ticker: str,
    period: str = PRICE_PERIOD,
    force_refresh: bool = False,
) -> pd.DataFrame:

    ticker = ticker.upper()

    cache_file = PRICE_CACHE_DIR / f"{ticker}_{period}.csv"

    if cache_file.exists() and not force_refresh:
        print(f"Loading cached prices for {ticker}...")
        return pd.read_csv(cache_file, parse_dates=["date"])

    print(f"Downloading prices for {ticker}...")
    step_t0 = perf_counter()

    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    print(
        f"[timing] yfinance download {ticker}: "
        f"{perf_counter() - step_t0:.2f}s"
    )

    if data.empty:
        raise ValueError(f"No price data for {ticker}")

    data = flatten_yfinance_columns(data)
    data = data.reset_index()
    data = flatten_yfinance_columns(data)

    data = data.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Adj_Close": "adj_close",
            "Volume": "volume",
        }
    )

    if "adj_close" not in data.columns and "close" in data.columns:
        data["adj_close"] = data["close"]

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            f"Missing columns for {ticker}: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data["ticker"] = ticker

    data = data[
        [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    ]

    data.to_csv(cache_file, index=False)

    return data


# =========================================================
# MARKET CONTEXT CACHE
# =========================================================

def load_market_context_cached(
    force_refresh: bool = False,
) -> pd.DataFrame:

    cache_file = MARKET_CACHE_DIR / "market_context.csv"

    if cache_file.exists() and not force_refresh:
        print("Loading cached market context...")
        return pd.read_csv(cache_file, parse_dates=["date"])

    print("Building market context...")
    step_t0 = perf_counter()

    market_context = build_market_context()

    print(
        f"[timing] build_market_context: "
        f"{perf_counter() - step_t0:.2f}s"
    )

    market_context.to_csv(cache_file, index=False)

    return market_context


# =========================================================
# GET LIVE NEWS FROM FINNHUB API
# =========================================================

def fetch_finnhub_company_news(
    ticker: str,
    days_back: int = NEWS_HISTORY_DAYS,
    force_refresh: bool = False,
) -> pd.DataFrame:

    ticker = ticker.upper()

    csv_cache_file = NEWS_CACHE_DIR / f"{ticker}_news.csv"
    raw_cache_file = NEWS_CACHE_DIR / f"{ticker}_raw_news.json"

    # 1. Prefer processed CSV cache, unless forced refresh
    if csv_cache_file.exists() and not force_refresh:
        print(f"Loading cached processed news for {ticker}...")
        cached = pd.read_csv(csv_cache_file, parse_dates=["date"])

        if "news_status" not in cached.columns:
            cached["news_status"] = "cached"

        return cached

    # 2. Try Supabase shared news cache
    try:
        print(f"Loading news from Supabase for {ticker}...")
        supabase_news = get_company_news_from_supabase(
            ticker=ticker,
            days_back=days_back,
            limit=50,
        )

        if not supabase_news.empty:
            supabase_news = (
                supabase_news
                .drop_duplicates(subset=["ticker", "date", "text"])
                .sort_values(["date"], ascending=False)
                .head(50)
                .sort_values(["ticker", "date"])
                .reset_index(drop=True)
            )

            supabase_news.to_csv(csv_cache_file, index=False)

            print("News rows from Supabase:", len(supabase_news))

            return supabase_news

    except Exception as exc:
        print(f"WARNING: failed to load news from Supabase for {ticker}: {exc}")

    # 3. Fallback to raw Finnhub JSON cache
    if raw_cache_file.exists() and not force_refresh:
        print(f"Loading cached raw news for {ticker}...")

        try:
            with open(raw_cache_file, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"WARNING: failed to read raw news cache for {ticker}: {exc}")
            return make_fallback_news_row(ticker, status="fallback")

        if not isinstance(data, list):
            print(f"WARNING: unexpected raw news format for {ticker}: {data}")
            return make_fallback_news_row(ticker, status="fallback")

        rows = []

        for item in data:
            headline = item.get("headline") or ""
            summary = item.get("summary") or ""

            text = f"{headline}. {summary}".strip(". ").strip()

            if not text:
                continue

            timestamp = item.get("datetime")

            if timestamp:
                news_date = pd.to_datetime(timestamp, unit="s").normalize()
            else:
                news_date = pd.Timestamp.today().normalize()

            rows.append(
                {
                    "ticker": ticker,
                    "date": news_date,
                    "text": text,
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "news_status": "live",
                }
            )

        news_df = pd.DataFrame(rows)

        if news_df.empty:
            return make_fallback_news_row(ticker, status="synthetic")

        news_df = (
            news_df
            .drop_duplicates(subset=["ticker", "date", "text"])
            .sort_values(["date"], ascending=False)
            .head(50)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

        news_df.to_csv(csv_cache_file, index=False)

        print("News rows from raw cache:", len(news_df))

        return news_df

    # 4. Last resort: direct Finnhub fetch from API container
    print(f"No shared/cache news found for {ticker}. Trying Finnhub live...")

    api_key = os.getenv("FINNHUB_API_KEY")

    if not api_key:
        print("WARNING: FINNHUB_API_KEY is not set.")
        return make_fallback_news_row(ticker, status="fallback")

    to_date = date.today()
    from_date = to_date - timedelta(days=days_back)

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

    except Exception as exc:
        print(f"WARNING: Finnhub request failed for {ticker}: {exc}")
        return make_fallback_news_row(ticker, status="fallback")

    if not isinstance(data, list):
        print(f"WARNING: unexpected Finnhub response for {ticker}: {data}")
        return make_fallback_news_row(ticker, status="fallback")

    raw_cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_cache_file, "w") as f:
        json.dump(data, f)

    # Re-enter function and parse raw cache
    return fetch_finnhub_company_news(
        ticker=ticker,
        days_back=days_back,
        force_refresh=False,
    )


# FINBERT SCORING
# =========================================================

def score_news(news_df: pd.DataFrame) -> pd.DataFrame:

    print("Scoring news with FinBERT...")
    step_t0 = perf_counter()

    texts = news_df["text"].fillna("").astype(str).tolist()

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = finbert(**enc).logits.detach().cpu().numpy()

    probs = softmax(logits, axis=1)
    pred_ids = probs.argmax(axis=1)

    label_map = {
        0: "positive",
        1: "negative",
        2: "neutral",
    }

    news_df = news_df.copy()
    news_df["sentiment_label"] = [label_map[pred] for pred in pred_ids]
    news_df["sentiment_score"] = [float(p[0] - p[1]) for p in probs]
    news_df["neg_prob"] = [float(p[1]) for p in probs]

    news_df["is_negative"] = (
        news_df["sentiment_label"] == "negative"
    ).astype(int)

    news_df["is_very_negative"] = (
        news_df["neg_prob"] >= 0.80
    ).astype(int)

    print(f"[timing] FinBERT scoring: {perf_counter() - step_t0:.2f}s")

    return news_df


# =========================================================
# BUILD LIVE FEATURES
# =========================================================

def build_live_features(
    ticker: str,
    force_refresh_prices: bool = False,
    force_refresh_news: bool = False,
    force_refresh_market: bool = False,
):

    log_time("build_live_features start")

    price_df = download_live_price_data(
        ticker,
        force_refresh=force_refresh_prices,
    )
    log_time("price ready")

    market_context = load_market_context_cached(
        force_refresh=force_refresh_market,
    )
    log_time("market context ready")

    step_t0 = perf_counter()
    price_df = add_market_context(
        price_df=price_df,
        market_context=market_context,
    )
    print(f"[timing] add_market_context: {perf_counter() - step_t0:.2f}s")

    step_t0 = perf_counter()
    price_features = add_price_features(price_df)
    print(f"[timing] add_price_features: {perf_counter() - step_t0:.2f}s")

    log_time("price features ready")

    news_df = fetch_finnhub_company_news(
        ticker,
        force_refresh=force_refresh_news,
    )
    log_time("news ready")

    news_df = score_news(news_df)
    log_time("news scored")

    step_t0 = perf_counter()
    news_features = add_news_features(news_df)
    print(f"[timing] add_news_features: {perf_counter() - step_t0:.2f}s")

    log_time("news features ready")

    full_df = price_features.merge(
        news_features,
        on=["ticker", "date"],
        how="left",
    )

    full_df = full_df.sort_values("date")
    latest = full_df.tail(1).copy()

    news_status = "unknown"

    if "news_status" in news_df.columns:
        news_status = str(news_df["news_status"].iloc[0])

    latest["news_status"] = news_status

    news_cols = [
        c for c in latest.columns
        if any(k in c.lower() for k in ["news", "sentiment", "neg"])
    ]

    print("\nDEBUG latest date:", latest["date"].iloc[0])
    print("DEBUG news_status:", news_status)
    print("DEBUG news feature columns:", len(news_cols))

    if news_cols:
        print(latest[news_cols].T.head(80))

    for col in FEATURE_COLUMNS:
        if col not in latest.columns:
            latest[col] = 0

    X = latest[FEATURE_COLUMNS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    log_time("features ready")

    return latest, X


# =========================================================
# PREDICT
# =========================================================

def predict_ticker_risk(
    ticker: str,
    force_refresh_prices: bool = False,
    force_refresh_news: bool = False,
    force_refresh_market: bool = False,
    debug_shap: bool = False,
):

    log_time("prediction pipeline start")

    latest, X = build_live_features(
        ticker,
        force_refresh_prices=force_refresh_prices,
        force_refresh_news=force_refresh_news,
        force_refresh_market=force_refresh_market,
    )

    step_t0 = perf_counter()

    probability = float(model.predict_proba(X)[0][1])

    print(f"[timing] model.predict_proba: {perf_counter() - step_t0:.2f}s")
    log_time("raw prediction ready")

    if debug_shap:
        print("\nTOP FEATURE CONTRIBUTIONS:")

        try:
            import shap

            step_t0 = perf_counter()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_row = shap_values[1][0]
            else:
                shap_row = shap_values[0]

            contributions = pd.DataFrame({
                "feature": FEATURE_COLUMNS,
                "value": X.iloc[0].values,
                "shap": shap_row,
            })

            contributions["abs_shap"] = contributions["shap"].abs()

            contributions = contributions.sort_values(
                "abs_shap",
                ascending=False,
            )

            print(
                contributions[
                    ["feature", "value", "shap"]
                ].head(25)
            )

            print(f"[timing] SHAP debug: {perf_counter() - step_t0:.2f}s")

        except Exception as exc:
            print("SHAP failed:", exc)

    step_t0 = perf_counter()

    result = generate_insight(
        ticker=ticker,
        probability=probability,
        threshold=THRESHOLD,
        latest_row=latest.iloc[0],
    )

    if "news_status" in latest.columns:
        result["news_status"] = str(latest["news_status"].iloc[0])

    print(f"[timing] generate_insight: {perf_counter() - step_t0:.2f}s")
    log_time("prediction done")

    return result

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--force-refresh-news", action="store_true")
    parser.add_argument("--force-refresh-prices", action="store_true")
    parser.add_argument("--force-refresh-market", action="store_true")
    parser.add_argument("--force-refresh-all", action="store_true")
    parser.add_argument("--debug-shap", action="store_true")

    args = parser.parse_args()

    force_all = args.force_refresh_all

    result = predict_ticker_risk(
        args.ticker.upper(),
        force_refresh_prices=force_all or args.force_refresh_prices,
        force_refresh_news=force_all or args.force_refresh_news,
        force_refresh_market=force_all or args.force_refresh_market,
        debug_shap=args.debug_shap,
    )

    print("\nRESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    log_time("total")