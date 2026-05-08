# src/sellsmart_ml/dataset/build_full_model_dataset.py

from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PRICE_FEATURES_PATH = Path(
    "data/processed/price_features.csv"
)

NEWS_FEATURES_PATH = Path(
    "data/processed/news_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/full_model_dataset.csv"
)


# =========================================================
# TARGETS
# =========================================================

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker", group_keys=False)

    # -----------------------------------------------------
    # FUTURE RETURNS
    # -----------------------------------------------------

    df["future_return_1d"] = (
        g["close"]
        .shift(-1) / df["close"] - 1
    )

    df["future_return_3d"] = (
        g["close"]
        .shift(-3) / df["close"] - 1
    )

    df["future_return_5d"] = (
        g["close"]
        .shift(-5) / df["close"] - 1
    )

    # -----------------------------------------------------
    # FUTURE MIN LOWS
    # -----------------------------------------------------

    df["future_min_low_3d"] = (
        g["low"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(3, min_periods=1)
                .min()
            )
        )
    )

    df["future_min_low_5d"] = (
        g["low"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(5, min_periods=1)
                .min()
            )
        )
    )

    # -----------------------------------------------------
    # DRAWDOWNS
    # -----------------------------------------------------

    df["future_drawdown_3d"] = (
        df["future_min_low_3d"] / df["close"] - 1
    )

    df["future_drawdown_5d"] = (
        df["future_min_low_5d"] / df["close"] - 1
    )

    # -----------------------------------------------------
    # TARGETS
    # -----------------------------------------------------

    df["target_drop_1d_3pct"] = (
        df["future_return_1d"] <= -0.03
    ).astype(int)

    df["target_drop_3d_4pct"] = (
        df["future_return_3d"] <= -0.04
    ).astype(int)

    df["target_drop_5d_5pct"] = (
        df["future_return_5d"] <= -0.05
    ).astype(int)

    df["target_drawdown_5d_7pct"] = (
        df["future_drawdown_5d"] <= -0.07
    ).astype(int)

    # -----------------------------------------------------
    # VOLATILITY SPIKE
    # -----------------------------------------------------

    future_volatility = (
        g["ret_1"]
        .transform(
            lambda x: (
                x.shift(-1)
                .rolling(3, min_periods=1)
                .std()
            )
        )
    )

    current_volatility = (
        g["ret_1"]
        .transform(
            lambda x: (
                x.rolling(20, min_periods=5)
                .std()
            )
        )
    )

    df["target_vol_spike_3d"] = (
        future_volatility >
        current_volatility * 1.8
    ).astype(int)

    return df


# =========================================================
# MAIN
# =========================================================

def build_full_model_dataset():

    print(f"Loading: {PRICE_FEATURES_PATH}")
    price_df = pd.read_csv(PRICE_FEATURES_PATH)

    print(f"Loading: {NEWS_FEATURES_PATH}")
    news_df = pd.read_csv(NEWS_FEATURES_PATH)

    # -----------------------------------------------------
    # NORMALIZE DATES
    # -----------------------------------------------------

    price_df["date"] = (
        pd.to_datetime(price_df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    news_df["date"] = (
        pd.to_datetime(news_df["date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    price_df["ticker"] = (
        price_df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    news_df["ticker"] = (
        news_df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    print("\nPRICE:", price_df.shape)
    print("NEWS: ", news_df.shape)

    # -----------------------------------------------------
    # MERGE
    # -----------------------------------------------------

    full_df = price_df.merge(
        news_df,
        on=["ticker", "date"],
        how="left"
    )

    # -----------------------------------------------------
    # FILL MISSING NEWS
    # -----------------------------------------------------

    numeric_cols = full_df.select_dtypes(
        include=[np.number]
    ).columns

    news_numeric_cols = [
        c for c in numeric_cols
        if (
            "news" in c
            or "sentiment" in c
            or "neg_" in c
            or "panic" in c
            or "event" in c
            or "downgrade" in c
            or "guidance" in c
            or "lawsuit" in c
            or "investigation" in c
            or "earnings" in c
            or "analyst" in c
            or "macro" in c
            or "severity" in c
            or "spike" in c
        )
    ]

    full_df[news_numeric_cols] = (
        full_df[news_numeric_cols]
        .fillna(0)
    )

    if "has_news" in full_df.columns:
        full_df["has_news"] = (
            full_df["has_news"]
            .fillna(0)
            .astype(int)
        )

    # -----------------------------------------------------
    # TARGETS
    # -----------------------------------------------------

    full_df = full_df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    full_df = add_targets(full_df)

    # -----------------------------------------------------
    # CLEAN
    # -----------------------------------------------------

    full_df = full_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    full_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFULL DATASET:", full_df.shape)

    target_cols = [
        "target_drop_1d_3pct",
        "target_drop_3d_4pct",
        "target_drop_5d_5pct",
        "target_drawdown_5d_7pct",
        "target_vol_spike_3d",
    ]

    print("\n===== TARGET EVENT RATES =====")

    for col in target_cols:
        print(
            f"{col}: "
            f"{full_df[col].mean():.4f}"
        )

    print("\nSaved merged dataset ->")
    print(OUTPUT_PATH)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    build_full_model_dataset()