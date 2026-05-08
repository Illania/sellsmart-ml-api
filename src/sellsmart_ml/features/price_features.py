from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std()


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_drawdown(close: pd.Series, window: int) -> pd.Series:
    rolling_max = close.rolling(window).max()
    return close / rolling_max - 1.0


def add_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    required = [
        "date", "ticker",
        "open", "high", "low", "close", "volume",
        "SPY_return_1d", "SPY_return_5d",
        "QQQ_return_1d", "QQQ_return_5d",
        "VIX", "VIX_change_1d", "VIX_spike_5d",
        "market_volatility_20d", "nasdaq_volatility_20d",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for price features: {missing}")

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().sort_values("date")

        close = g["close"]
        volume = g["volume"]
        daily_ret = close.pct_change()

        # returns
        g["ret_1"] = close.pct_change(1)
        g["ret_2"] = close.pct_change(2)
        g["ret_3"] = close.pct_change(3)
        g["ret_5"] = close.pct_change(5)
        g["ret_10"] = close.pct_change(10)
        g["ret_20"] = close.pct_change(20)

        # volatility
        g["vol_5"] = rolling_volatility(daily_ret, 5)
        g["vol_10"] = rolling_volatility(daily_ret, 10)
        g["vol_20"] = rolling_volatility(daily_ret, 20)

        # moving averages
        g["ma_10"] = close.rolling(10).mean()
        g["ma_20"] = close.rolling(20).mean()
        g["ma_50"] = close.rolling(50).mean()

        g["dist_ma_10"] = safe_div(close, g["ma_10"]) - 1.0
        g["dist_ma_20"] = safe_div(close, g["ma_20"]) - 1.0
        g["dist_ma_50"] = safe_div(close, g["ma_50"]) - 1.0

        # RSI
        g["rsi_14"] = compute_rsi(close, 14)
        g["rsi_change_1d"] = g["rsi_14"].diff(1)

        # volume
        vol_ma_5 = volume.rolling(5).mean()
        vol_ma_20 = volume.rolling(20).mean()

        g["vol_ratio_5"] = safe_div(volume, vol_ma_5)
        g["vol_ratio_20"] = safe_div(volume, vol_ma_20)
        g["volume_ret_1"] = volume.pct_change(1)

        # drawdown
        g["drawdown_20"] = compute_drawdown(close, 20)
        g["drawdown_50"] = compute_drawdown(close, 50)

        # candle / intraday features
        g["high_low_range_1d"] = safe_div(g["high"] - g["low"], close)
        g["close_open_change"] = safe_div(g["close"] - g["open"], g["open"])

        # z-score / trend regime
        close_mean_20 = close.rolling(20).mean()
        close_std_20 = close.rolling(20).std()

        g["price_zscore_20"] = (
            (close - close_mean_20) /
            close_std_20.replace(0, np.nan)
        )

        g["trend_strength_20"] = safe_div(g["ma_10"] - g["ma_20"], g["ma_20"])
        g["trend_strength_50"] = safe_div(g["ma_20"] - g["ma_50"], g["ma_50"])

        vol20_med_60 = g["vol_20"].rolling(60).median()
        g["volatility_regime"] = (g["vol_20"] > vol20_med_60).astype(float)

        return g

    frames = []

    for ticker, g in df.groupby("ticker", group_keys=False):
        processed = _per_ticker(g)
        processed["ticker"] = ticker
        frames.append(processed)

    df = pd.concat(frames, ignore_index=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def get_final_price_feature_columns() -> list[str]:
    return [
        "open", "high", "low", "close", "volume",

        "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",

        "vol_5", "vol_10", "vol_20",

        "ma_10", "ma_20", "ma_50",

        "dist_ma_10", "dist_ma_20", "dist_ma_50",

        "rsi_14", "rsi_change_1d",

        "vol_ratio_5", "vol_ratio_20", "volume_ret_1",

        "drawdown_20", "drawdown_50",

        "high_low_range_1d", "close_open_change",
        "price_zscore_20",
        "trend_strength_20", "trend_strength_50",
        "volatility_regime",

        "SPY_return_1d", "SPY_return_5d",
        "QQQ_return_1d", "QQQ_return_5d",

        "VIX", "VIX_change_1d", "VIX_spike_5d",

        "market_volatility_20d",
        "nasdaq_volatility_20d",
    ]


def validate_price_features(df: pd.DataFrame) -> None:
    required = get_final_price_feature_columns()

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing final price features: {missing}")

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found.")
