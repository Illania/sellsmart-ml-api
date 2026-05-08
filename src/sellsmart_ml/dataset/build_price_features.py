from __future__ import annotations

import pandas as pd

from sellsmart_ml.config import PROCESSED_DATA_DIR
from sellsmart_ml.features.price_features import (
    add_price_features,
    validate_price_features,
)


INPUT_FILE = PROCESSED_DATA_DIR / "price_dataset.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "price_features.csv"


def build_price_features() -> pd.DataFrame:
    print(f"Loading price dataset: {INPUT_FILE}")

    price_df = pd.read_csv(INPUT_FILE)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.tz_localize(None)

    features_df = add_price_features(price_df)
    validate_price_features(features_df)

    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved price features: {OUTPUT_FILE}")
    print("Shape:", features_df.shape)
    print("Date range:", features_df["date"].min(), "->", features_df["date"].max())
    print("Tickers:", sorted(features_df["ticker"].unique()))

    feature_cols = [
        c for c in features_df.columns
        if c not in ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    ]

    print("Feature columns:")
    for col in feature_cols:
        print(" -", col)

    return features_df


if __name__ == "__main__":
    build_price_features()
