from __future__ import annotations

import pandas as pd
import yfinance as yf

from sellsmart_ml.config import (
    TICKERS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)


START_DATE = "2009-08-17"
END_DATE = "2024-01-09"

RAW_OUTPUT_FILE = RAW_DATA_DIR / "price_raw.csv"
PROCESSED_OUTPUT_FILE = PROCESSED_DATA_DIR / "price_dataset.csv"


def _normalize_downloaded_price_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        raise ValueError(f"No data downloaded for ticker: {ticker}")

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[-1] for col in df.columns]

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[-1] for col in df.columns]

    rename_map = {
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

    df = df.rename(columns=rename_map)

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    required_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for {ticker}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker

    return df[
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


def download_single_ticker(
    ticker: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    yf_ticker = "META" if ticker == "FB" else ticker

    print(f"Downloading {ticker} from yfinance as {yf_ticker}...")

    data = yf.download(
        yf_ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    return _normalize_downloaded_price_data(data, ticker=ticker)


def download_price_data(
    tickers: list[str] | tuple[str, ...],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            frames.append(
                download_single_ticker(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception as exc:
            print(f"WARNING: failed to download {ticker}: {exc}")

    if not frames:
        raise RuntimeError("No ticker data was downloaded.")

    df = pd.concat(frames, ignore_index=True)

    df = (
        df.drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    return df


def build_market_context(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    market_tickers = ["SPY", "QQQ", "^VIX"]

    market_df = download_price_data(
        tickers=market_tickers,
        start_date=start_date,
        end_date=end_date,
    )

    spy = market_df[market_df["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date")
    spy["SPY_return_1d"] = spy["close"].pct_change(1)
    spy["SPY_return_5d"] = spy["close"].pct_change(5)
    spy["market_volatility_20d"] = spy["SPY_return_1d"].rolling(20).std()
    spy = spy.drop(columns=["close"])

    qqq = market_df[market_df["ticker"] == "QQQ"][["date", "close"]].copy()
    qqq = qqq.sort_values("date")
    qqq["QQQ_return_1d"] = qqq["close"].pct_change(1)
    qqq["QQQ_return_5d"] = qqq["close"].pct_change(5)
    qqq["nasdaq_volatility_20d"] = qqq["QQQ_return_1d"].rolling(20).std()
    qqq = qqq.drop(columns=["close"])

    vix = market_df[market_df["ticker"] == "^VIX"][["date", "close"]].copy()
    vix = vix.sort_values("date")
    vix = vix.rename(columns={"close": "VIX"})
    vix["VIX_change_1d"] = vix["VIX"].pct_change(1)
    vix["VIX_spike_5d"] = vix["VIX"] / vix["VIX"].rolling(5).mean() - 1

    market_context = spy.merge(qqq, on="date", how="outer")
    market_context = market_context.merge(vix, on="date", how="outer")

    return market_context.sort_values("date").reset_index(drop=True)


def add_market_context(
    price_df: pd.DataFrame,
    market_context: pd.DataFrame,
) -> pd.DataFrame:
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    market_context = market_context.copy()
    market_context["date"] = pd.to_datetime(market_context["date"]).dt.tz_localize(None)

    df = df.merge(market_context, on="date", how="left")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


def validate_price_dataset(df: pd.DataFrame) -> None:
    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "SPY_return_1d",
        "SPY_return_5d",
        "QQQ_return_1d",
        "QQQ_return_5d",
        "VIX",
        "VIX_change_1d",
        "VIX_spike_5d",
        "market_volatility_20d",
        "nasdaq_volatility_20d",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Price dataset missing columns: {missing}")

    if df[["date", "ticker"]].duplicated().any():
        duplicates = df[df[["date", "ticker"]].duplicated(keep=False)]
        raise ValueError(
            "Duplicate rows found for date/ticker pairs. "
            f"Example duplicates:\n{duplicates.head()}"
        )

    if df["close"].isna().all():
        raise ValueError("All close values are NaN.")

    if df["ticker"].nunique() == 0:
        raise ValueError("No tickers found in price dataset.")


def build_price_dataset(
    tickers: list[str] | tuple[str, ...] = TICKERS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    save: bool = True,
) -> pd.DataFrame:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    price_df = download_price_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )

    if save:
        price_df.to_csv(RAW_OUTPUT_FILE, index=False)
        print(f"Saved raw price data: {RAW_OUTPUT_FILE}")

    market_context = build_market_context(
        start_date=start_date,
        end_date=end_date,
    )

    price_df = add_market_context(
        price_df=price_df,
        market_context=market_context,
    )

    validate_price_dataset(price_df)

    if save:
        price_df.to_csv(PROCESSED_OUTPUT_FILE, index=False)
        print(f"Saved processed price dataset: {PROCESSED_OUTPUT_FILE}")

    print("Price dataset shape:", price_df.shape)
    print("Date range:", price_df["date"].min(), "->", price_df["date"].max())
    print("Tickers:", sorted(price_df["ticker"].unique()))

    return price_df


if __name__ == "__main__":
    build_price_dataset()