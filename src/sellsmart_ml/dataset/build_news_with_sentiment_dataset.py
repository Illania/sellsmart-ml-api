# src/sellsmart_ml/dataset/build_news_with_sentiment_dataset.py

# =========================================================
# BUILD ARTICLE-LEVEL NEWS DATASET WITH FINBERT SENTIMENT
#
# Output:
#   data/raw/news_articles_with_sentiment.csv
#
# Source:
#   HuggingFace dataset:
#   benstaf/FNSPID-filtered-nasdaq-100
#
# =========================================================

import pandas as pd
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from scipy.special import softmax
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "NFLX", "JPM",
    "CRM", "ADBE", "INTC", "QCOM", "PYPL"
]

HF_DATASET_NAME = "benstaf/FNSPID-filtered-nasdaq-100"

MAX_MATCHED_ROWS = 200_000

OUTPUT_PATH = "data/raw/news_articles_with_sentiment.csv"

MODEL_NAME = "ProsusAI/finbert"

BATCH_SIZE = 64
MAX_LENGTH = 256


# =========================================================
# LOAD NEWS ARTICLES
# =========================================================

def load_news_articles():
    print("Streaming news dataset...")

    dataset = load_dataset(
        HF_DATASET_NAME,
        split="train",
        streaming=True
    )

    rows = []

    matched = 0
    scanned = 0

    for row in dataset:
        scanned += 1

        ticker = row.get("Stock_symbol")
        date = row.get("Date")

        if ticker not in TICKERS or date is None:
            if scanned % 50000 == 0:
                print(
                    f"Scanned {scanned:,} rows "
                    f"| matched {matched:,}"
                )
            continue

        try:
            d = pd.to_datetime(date).normalize()
        except Exception:
            continue

        text = (
            row.get("Article_title")
            or row.get("Title")
            or row.get("title")
            or row.get("article_title")
            or row.get("Headline")
            or row.get("headline")
            or ""
        )

        if not isinstance(text, str):
            text = str(text)

        text = text.strip()

        if not text:
            continue

        rows.append({
            "ticker": str(ticker).upper().strip(),
            "date": d,
            "text": text
        })

        matched += 1

        if matched % 1000 == 0:
            print(
                f"Matched {matched:,} rows "
                f"after scanning {scanned:,}"
            )

        if matched >= MAX_MATCHED_ROWS:
            break

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["ticker", "date", "text"]
    )

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    print("\nLoaded articles:")
    print(df.shape)

    return df


# =========================================================
# LOAD FINBERT
# =========================================================

def load_finbert():
    print("\nLoading FinBERT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    print("Device:", device)

    return tokenizer, model, device


# =========================================================
# SCORE TEXTS
# =========================================================

def score_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
):

    label_map = {
        0: "positive",
        1: "negative",
        2: "neutral"
    }

    all_labels = []
    all_scores = []
    all_neg_probs = []

    texts = [
        "" if pd.isna(x) else str(x)
        for x in texts
    ]

    total = len(texts)

    for i in range(0, total, batch_size):

        batch = texts[i:i + batch_size]

        print(
            f"Processing batch "
            f"{i:,} / {total:,}"
        )

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        with torch.no_grad():
            logits = (
                model(**enc)
                .logits
                .detach()
                .cpu()
                .numpy()
            )

        probs = softmax(logits, axis=1)

        pred_ids = probs.argmax(axis=1)

        for p, pred in zip(probs, pred_ids):

            label = label_map[pred]

            # signed sentiment:
            # positive -> positive value
            # negative -> negative value
            signed_score = float(p[0] - p[1])

            all_labels.append(label)

            all_scores.append(signed_score)

            all_neg_probs.append(float(p[1]))

    return (
        all_labels,
        all_scores,
        all_neg_probs
    )


# =========================================================
# BUILD DATASET
# =========================================================

def build_news_dataset():

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Load news
    # -----------------------------------------------------

    news_df = load_news_articles()

    # -----------------------------------------------------
    # Load FinBERT
    # -----------------------------------------------------

    tokenizer, model, device = load_finbert()

    # -----------------------------------------------------
    # Score sentiment
    # -----------------------------------------------------

    texts = news_df["text"].fillna("").tolist()

    labels, scores, neg_probs = score_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device
    )

    # -----------------------------------------------------
    # Save sentiment columns
    # -----------------------------------------------------

    news_df["sentiment_label"] = labels

    news_df["sentiment_score"] = scores

    news_df["neg_prob"] = neg_probs

    news_df["is_negative"] = (
        news_df["sentiment_label"] == "negative"
    ).astype(int)

    news_df["is_very_negative"] = (
        news_df["neg_prob"] >= 0.80
    ).astype(int)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    news_df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved dataset:")
    print(output_path)

    print("\nFinal shape:")
    print(news_df.shape)

    print("\nSample:")
    print(news_df.head())


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    build_news_dataset()