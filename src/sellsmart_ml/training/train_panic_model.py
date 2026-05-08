# src/sellsmart_ml/training/train_panic_model.py

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)


# =========================================================
# CONFIG
# =========================================================

DATA_PATH = Path("data/processed/full_model_dataset.csv")

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET_COL = "target_drawdown_5d_7pct"

TRAIN_END = "2022-06-30"
VALID_END = "2023-06-30"

RANDOM_STATE = 42


# =========================================================
# FEATURE GROUPS
# =========================================================

PRICE_FEATURE_KEYWORDS = [
    "return",
    "ret_",
    "vol_",
    "volatility",
    "drawdown",
    "ma_",
    "dist_ma",
    "rsi",
    "volume",
    "vol_ratio",
    "high_low",
    "spy",
    "qqq",
    "vix",
    "market",
    "nasdaq",
    "trend",
]

NEWS_FEATURE_KEYWORDS = [
    "news",
    "sentiment",
    "neg_",
    "negative",
    "panic",
    "event",
    "downgrade",
    "upgrade",
    "guidance",
    "lawsuit",
    "investigation",
    "earnings",
    "analyst",
    "macro",
    "severity",
    "spike",
    "bankruptcy",
    "product",
    "mna",
]


LEAKAGE_KEYWORDS = [
    "future",
    "target",
]


ID_COLS = [
    "date",
    "ticker",
]


# =========================================================
# UTILS
# =========================================================

def is_numeric_feature(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])


def contains_any(col: str, keywords: list[str]) -> bool:
    col_lc = col.lower()
    return any(k in col_lc for k in keywords)


def build_feature_sets(df: pd.DataFrame):
    numeric_cols = [
        c for c in df.columns
        if is_numeric_feature(df, c)
    ]

    usable_cols = []

    for c in numeric_cols:
        c_lc = c.lower()

        if c in ID_COLS:
            continue

        if any(k in c_lc for k in LEAKAGE_KEYWORDS):
            continue

        usable_cols.append(c)

    price_features = [
        c for c in usable_cols
        if contains_any(c, PRICE_FEATURE_KEYWORDS)
        and not contains_any(c, NEWS_FEATURE_KEYWORDS)
    ]

    news_features = [
        c for c in usable_cols
        if contains_any(c, NEWS_FEATURE_KEYWORDS)
    ]

    price_plus_news_features = sorted(
        list(set(price_features + news_features))
    )

    price_features = sorted(price_features)
    news_features = sorted(news_features)

    return price_features, news_features, price_plus_news_features


def time_split(df: pd.DataFrame):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] <= TRAIN_END].copy()

    valid_df = df[
        (df["date"] > TRAIN_END) &
        (df["date"] <= VALID_END)
    ].copy()

    test_df = df[df["date"] > VALID_END].copy()

    return train_df, valid_df, test_df


def clean_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target_col: str,
):
    keep_cols = features + [target_col]

    train = train_df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid_df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    test = test_df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()

    X_train = train[features]
    y_train = train[target_col].astype(int)

    X_valid = valid[features]
    y_valid = valid[target_col].astype(int)

    X_test = test[features]
    y_test = test[target_col].astype(int)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = int(len(y) - pos)

    if pos == 0:
        return 1.0

    return neg / pos


def make_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )


def evaluate_predictions(
    y_true,
    proba,
    threshold: float = 0.5,
):
    pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "event_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "threshold": threshold,
    }

    return metrics


def topk_event_rates(y_true, proba, ks=(0.05, 0.10, 0.20)):
    result = {}

    tmp = pd.DataFrame({
        "y": y_true,
        "proba": proba,
    }).sort_values("proba", ascending=False)

    overall = tmp["y"].mean()

    result["overall_event_rate"] = overall

    for k in ks:
        n = max(1, int(len(tmp) * k))
        top_rate = tmp.head(n)["y"].mean()

        result[f"top_{int(k * 100)}pct_event_rate"] = top_rate
        result[f"top_{int(k * 100)}pct_lift"] = (
            top_rate / overall if overall > 0 else np.nan
        )

    return result


def find_best_threshold(y_true, proba):
    rows = []

    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "predicted_positive_rate": float(np.mean(pred)),
        })

    df = pd.DataFrame(rows)

    best = df.sort_values("f1", ascending=False).iloc[0]

    return float(best["threshold"]), df


def train_one_model(
    name: str,
    features: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("Features:", len(features))
    print("=" * 80)

    X_train, y_train, X_valid, y_valid, X_test, y_test = clean_xy(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        features=features,
        target_col=TARGET_COL,
    )

    print("Train:", X_train.shape, "| event rate:", round(y_train.mean(), 4))
    print("Valid:", X_valid.shape, "| event rate:", round(y_valid.mean(), 4))
    print("Test: ", X_test.shape, "| event rate:", round(y_test.mean(), 4))

    scale_pos_weight = get_scale_pos_weight(y_train)

    print("scale_pos_weight:", round(scale_pos_weight, 3))

    model = make_model(scale_pos_weight)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    best_threshold, threshold_df = find_best_threshold(
        y_valid,
        valid_proba,
    )

    print("Best valid threshold:", round(best_threshold, 3))

    valid_metrics = evaluate_predictions(
        y_valid,
        valid_proba,
        threshold=best_threshold,
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_proba,
        threshold=best_threshold,
    )

    valid_topk = topk_event_rates(y_valid, valid_proba)
    test_topk = topk_event_rates(y_test, test_proba)

    print("\nVALID METRICS")
    print(valid_metrics)
    print(valid_topk)

    print("\nTEST METRICS")
    print(test_metrics)
    print(test_topk)

    test_pred = (test_proba >= best_threshold).astype(int)

    print("\nTEST CONFUSION MATRIX")
    print(confusion_matrix(y_test, test_pred))

    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(y_test, test_pred, zero_division=0))

    feature_importance = (
        pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    model_path = MODEL_DIR / f"panic_model_{name}.pkl"
    joblib.dump(model, model_path)

    feature_path = MODEL_DIR / f"panic_model_{name}_features.json"
    with open(feature_path, "w") as f:
        json.dump(features, f, indent=2)

    threshold_path = MODEL_DIR / f"panic_model_{name}_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": best_threshold}, f, indent=2)

    threshold_df.to_csv(
        REPORT_DIR / f"panic_model_{name}_thresholds.csv",
        index=False,
    )

    feature_importance.to_csv(
        REPORT_DIR / f"panic_model_{name}_feature_importance.csv",
        index=False,
    )

    result = {
        "model": name,
        "features": len(features),
        "best_threshold": best_threshold,

        "valid_roc_auc": valid_metrics["roc_auc"],
        "valid_pr_auc": valid_metrics["pr_auc"],
        "valid_f1": valid_metrics["f1"],
        "valid_precision": valid_metrics["precision"],
        "valid_recall": valid_metrics["recall"],
        "valid_event_rate": valid_metrics["event_rate"],

        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_event_rate": test_metrics["event_rate"],

        "test_top_5pct_event_rate": test_topk["top_5pct_event_rate"],
        "test_top_10pct_event_rate": test_topk["top_10pct_event_rate"],
        "test_top_20pct_event_rate": test_topk["top_20pct_event_rate"],

        "test_top_5pct_lift": test_topk["top_5pct_lift"],
        "test_top_10pct_lift": test_topk["top_10pct_lift"],
        "test_top_20pct_lift": test_topk["top_20pct_lift"],
    }

    topk_rows = []

    for split_name, topk in [
        ("valid", valid_topk),
        ("test", test_topk),
    ]:
        row = {
            "model": name,
            "split": split_name,
        }
        row.update(topk)
        topk_rows.append(row)

    return result, topk_rows


# =========================================================
# MAIN
# =========================================================

def train_panic_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COL}. "
            f"Available target columns: "
            f"{[c for c in df.columns if c.startswith('target_')]}"
        )

    price_features, news_features, price_plus_news_features = build_feature_sets(df)

    print("\nFeature groups:")
    print("price_features:", len(price_features))
    print("news_features:", len(news_features))
    print("price_plus_news_features:", len(price_plus_news_features))

    if len(price_features) == 0:
        raise ValueError("No price features found.")

    if len(news_features) == 0:
        raise ValueError("No news features found.")

    train_df, valid_df, test_df = time_split(df)

    print("\nSplit sizes before dropna:")
    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test: ", test_df.shape)

    results = []
    topk_results = []

    for name, features in [
        ("price_only", price_features),
        ("price_plus_news", price_plus_news_features),
    ]:
        result, topk_rows = train_one_model(
            name=name,
            features=features,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
        )

        results.append(result)
        topk_results.extend(topk_rows)

    comparison_df = pd.DataFrame(results)

    comparison_path = REPORT_DIR / "panic_model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    topk_df = pd.DataFrame(topk_results)

    topk_path = REPORT_DIR / "panic_model_topk.csv"

    topk_df.to_csv(
        topk_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df)

    print("\nSaved:")
    print(comparison_path)
    print(topk_path)
    print(MODEL_DIR / "panic_model_price_only.pkl")
    print(MODEL_DIR / "panic_model_price_plus_news.pkl")


if __name__ == "__main__":
    train_panic_model()