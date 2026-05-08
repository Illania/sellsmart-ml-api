from __future__ import annotations

import pandas as pd

from sellsmart_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from sellsmart_ml.features.news_features import (
    add_news_features,
    validate_news_features,
)


INPUT_FILE = RAW_DATA_DIR / "news_articles_with_sentiment.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "news_features.csv"


def build_news_features() -> pd.DataFrame:
    print(f"Loading news dataset: {INPUT_FILE}")

    news_articles = pd.read_csv(INPUT_FILE)

    features_df = add_news_features(news_articles)
    validate_news_features(features_df)

    features_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved news features: {OUTPUT_FILE}")
    print("Shape:", features_df.shape)
    print("Date range:", features_df["date"].min(), "->", features_df["date"].max())
    print("Tickers:", sorted(features_df["ticker"].unique()))
    print("Columns:", len(features_df.columns))

    return features_df


if __name__ == "__main__":
    build_news_features()