from __future__ import annotations

import re
import numpy as np
import pandas as pd


EVENT_PATTERNS = {
    "earnings": [
        r"\bearnings\b", r"\brevenue\b", r"\beps\b", r"\bfiscal\b",
        r"\bquarter(ly)? results?\b", r"\bprofit\b",
        r"\bmiss(es|ed)? estimates?\b", r"\bbeat(s|ing)? estimates?\b",
    ],
    "guidance_cut": [
        r"\bcuts? guidance\b", r"\blowers? guidance\b", r"\bweak guidance\b",
        r"\breduces? forecast\b", r"\btrims? outlook\b", r"\bslashes? forecast\b",
    ],
    "guidance_raise": [
        r"\braises? guidance\b", r"\bboosts? guidance\b",
        r"\bimproves? outlook\b", r"\braises? forecast\b", r"\bstrong guidance\b",
    ],
    "downgrade": [
        r"\bdowngrade\b", r"\bdowngraded\b", r"\bcut to sell\b",
        r"\bcut to underperform\b", r"\blowered rating\b", r"\brating cut\b",
    ],
    "upgrade": [
        r"\bupgrade\b", r"\bupgraded\b", r"\braised to buy\b",
        r"\braised rating\b", r"\bboosted rating\b", r"\brating raised\b",
    ],
    "lawsuit": [
        r"\blawsuit\b", r"\bsues?\b", r"\bsued\b", r"\blegal action\b",
        r"\bsettlement\b", r"\bcourt\b", r"\bclass action\b",
    ],
    "investigation": [
        r"\binvestigation\b", r"\bprobe\b", r"\bregulator\b",
        r"\bsubpoena\b", r"\bantitrust\b", r"\bsec\b", r"\bdoj\b",
    ],
    "product": [
        r"\blaunch\b", r"\brelease\b", r"\bunveil\b", r"\bintroduce\b",
        r"\bnew product\b", r"\bdevice\b", r"\bplatform\b", r"\bservice\b",
    ],
    "mna": [
        r"\bacquire\b", r"\bacquires\b", r"\bacquisition\b",
        r"\bmerger\b", r"\btakeover\b", r"\bbuyout\b",
    ],
    "bankruptcy": [
        r"\bbankrupt\b", r"\bbankruptcy\b", r"\binsolvency\b",
        r"\bchapter 11\b", r"\brestructuring\b",
    ],
    "analyst": [
        r"\banalyst\b", r"\bprice target\b", r"\bcoverage\b",
        r"\boutperform\b", r"\bunderperform\b", r"\boverweight\b",
        r"\bunderweight\b",
    ],
    "macro_risk": [
        r"\brecession\b", r"\binflation\b", r"\brate hike\b",
        r"\bslowdown\b", r"\bcrisis\b", r"\btariff\b", r"\bgeopolitical\b",
    ],
}


def normalize_news_input(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.copy()

    rename_map = {
        "Date": "date",
        "Stock_symbol": "ticker",
        "Article_title": "text",
        "Title": "text",
        "title": "text",
        "headline": "text",
        "Headline": "text",
        "sentiment": "sentiment_score",
        "negative_probability": "neg_prob",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["date", "ticker", "text"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw news columns: {missing}. Available: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    df = df[df["text"] != ""].copy()
    df = df.drop_duplicates(subset=["ticker", "date", "text"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "sentiment_score" not in df.columns:
        raise ValueError(
            "Missing sentiment_score. First run FinBERT sentiment scoring "
            "and save news_articles_with_sentiment.csv."
        )

    if "neg_prob" not in df.columns:
        raise ValueError(
            "Missing neg_prob. First run FinBERT sentiment scoring "
            "and save news_articles_with_sentiment.csv."
        )

    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = np.where(
            df["sentiment_score"] < -0.05,
            "negative",
            np.where(df["sentiment_score"] > 0.05, "positive", "neutral"),
        )

    if "is_negative" not in df.columns:
        df["is_negative"] = (df["sentiment_label"] == "negative").astype(int)

    if "is_very_negative" not in df.columns:
        df["is_very_negative"] = (
            (df["neg_prob"] >= 0.80) |
            (df["sentiment_score"] <= -0.75)
        ).astype(int)

    return df


def add_event_tags(news_articles: pd.DataFrame) -> pd.DataFrame:
    df = news_articles.copy()
    df["text_lc"] = df["text"].fillna("").astype(str).str.lower()

    for event_name, patterns in EVENT_PATTERNS.items():
        regex = "|".join(patterns)
        df[f"tag_{event_name}"] = (
            df["text_lc"].str.contains(regex, regex=True, na=False).astype(int)
        )

    df["strong_neg_article"] = (
        (df["is_negative"] == 1) &
        (df["neg_prob"] >= 0.80)
    ).astype(int)

    df["extreme_neg_article"] = (
        (df["is_very_negative"] == 1) |
        (df["sentiment_score"] <= -0.75) |
        (df["neg_prob"] >= 0.95)
    ).astype(int)

    df["positive_event_article"] = (
        df["tag_product"] |
        df["tag_upgrade"] |
        df["tag_guidance_raise"]
    ).astype(int)

    df["negative_event_article"] = (
        df["tag_downgrade"] |
        df["tag_guidance_cut"] |
        df["tag_lawsuit"] |
        df["tag_investigation"] |
        df["tag_bankruptcy"]
    ).astype(int)

    df["positive_event_negative_tone_article"] = (
        (df["positive_event_article"] == 1) &
        (
            (df["sentiment_score"] < 0) |
            (df["neg_prob"] > 0.6)
        )
    ).astype(int)

    return df


def add_news_features(news_articles: pd.DataFrame) -> pd.DataFrame:
    news_articles = normalize_news_input(news_articles)
    news_articles = add_event_tags(news_articles)

    daily_news = (
        news_articles
        .groupby(["ticker", "date"], as_index=False)
        .agg(
            news_count_1d=("text", "size"),
            neg_count_1d=("is_negative", "sum"),
            very_neg_count_1d=("is_very_negative", "sum"),
            sentiment_mean_1d=("sentiment_score", "mean"),
            sentiment_min_1d=("sentiment_score", "min"),
            sentiment_max_1d=("sentiment_score", "max"),
            sentiment_std_1d=("sentiment_score", "std"),
            neg_prob_mean_1d=("neg_prob", "mean"),

            strong_neg_count_1d=("strong_neg_article", "sum"),
            extreme_neg_count_1d=("extreme_neg_article", "sum"),
            positive_event_count_1d=("positive_event_article", "sum"),
            negative_event_count_1d=("negative_event_article", "sum"),
            positive_event_negative_tone_count_1d=("positive_event_negative_tone_article", "sum"),

            earnings_news_count_1d=("tag_earnings", "sum"),
            guidance_cut_news_count_1d=("tag_guidance_cut", "sum"),
            guidance_raise_news_count_1d=("tag_guidance_raise", "sum"),
            downgrade_news_count_1d=("tag_downgrade", "sum"),
            upgrade_news_count_1d=("tag_upgrade", "sum"),
            lawsuit_news_count_1d=("tag_lawsuit", "sum"),
            investigation_news_count_1d=("tag_investigation", "sum"),
            product_news_count_1d=("tag_product", "sum"),
            mna_news_count_1d=("tag_mna", "sum"),
            bankruptcy_news_count_1d=("tag_bankruptcy", "sum"),
            analyst_news_count_1d=("tag_analyst", "sum"),
            macro_risk_news_count_1d=("tag_macro_risk", "sum"),
        )
    )

    daily_news["has_news"] = 1
    daily_news["neg_ratio_1d"] = daily_news["neg_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["very_neg_ratio_1d"] = daily_news["very_neg_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["sentiment_std_1d"] = daily_news["sentiment_std_1d"].fillna(0)
    daily_news["sentiment_range_1d"] = daily_news["sentiment_max_1d"] - daily_news["sentiment_min_1d"]
    daily_news["news_count_log"] = np.log1p(daily_news["news_count_1d"])

    daily_news["strong_negative_article_ratio_1d"] = daily_news["strong_neg_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["extreme_negative_article_ratio_1d"] = daily_news["extreme_neg_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["positive_event_ratio_1d"] = daily_news["positive_event_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["negative_event_ratio_1d"] = daily_news["negative_event_count_1d"] / (daily_news["news_count_1d"] + 1e-9)
    daily_news["positive_event_negative_tone_ratio_1d"] = daily_news["positive_event_negative_tone_count_1d"] / (daily_news["news_count_1d"] + 1e-9)

    event_names = [
        "earnings", "guidance_cut", "guidance_raise", "downgrade", "upgrade",
        "lawsuit", "investigation", "product", "mna", "bankruptcy",
        "analyst", "macro_risk",
    ]

    for event_name in event_names:
        daily_news[f"has_{event_name}_news"] = (
            daily_news[f"{event_name}_news_count_1d"] > 0
        ).astype(int)

        daily_news[f"{event_name}_news_ratio_1d"] = (
            daily_news[f"{event_name}_news_count_1d"] /
            (daily_news["news_count_1d"] + 1e-9)
        )

    df = daily_news.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker", group_keys=False)

    for window in [3, 5, 7]:
        df[f"news_count_{window}d"] = g["news_count_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
        df[f"neg_count_{window}d"] = g["neg_count_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
        df[f"neg_ratio_{window}d"] = g["neg_ratio_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
        df[f"sentiment_mean_{window}d"] = g["sentiment_mean_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )
        df[f"sentiment_min_{window}d"] = g["sentiment_min_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).min()
        )
        df[f"sentiment_std_{window}d"] = g["sentiment_mean_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).std()
        ).fillna(0)

    for window in [3, 5]:
        df[f"very_neg_count_{window}d"] = g["very_neg_count_1d"].transform(
            lambda x, w=window: x.rolling(w, min_periods=1).mean()
        )

    event_count_cols = [
        "earnings_news_count_1d",
        "guidance_cut_news_count_1d",
        "guidance_raise_news_count_1d",
        "downgrade_news_count_1d",
        "upgrade_news_count_1d",
        "lawsuit_news_count_1d",
        "investigation_news_count_1d",
        "product_news_count_1d",
        "mna_news_count_1d",
        "bankruptcy_news_count_1d",
        "analyst_news_count_1d",
        "macro_risk_news_count_1d",
    ]

    extra_count_cols = [
        "strong_neg_count_1d",
        "extreme_neg_count_1d",
        "positive_event_count_1d",
        "negative_event_count_1d",
        "positive_event_negative_tone_count_1d",
    ]

    for col in event_count_cols + extra_count_cols:
        for window in [3, 7]:
            df[f"{col[:-3]}{window}d"] = g[col].transform(
                lambda x, w=window: x.rolling(w, min_periods=1).sum()
            )

    for col in ["news_count_1d", "neg_count_1d", "neg_ratio_1d", "sentiment_mean_1d"]:
        df[f"{col}_20d_avg"] = g[col].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

    for col in [
        "downgrade_news_count_1d",
        "guidance_cut_news_count_1d",
        "lawsuit_news_count_1d",
        "investigation_news_count_1d",
        "negative_event_count_1d",
    ]:
        df[f"{col}_20d_avg"] = g[col].transform(
            lambda x: x.shift(1).rolling(20, min_periods=3).mean()
        )

    df["news_count_20d_avg"] = df["news_count_1d_20d_avg"]
    df["neg_count_20d_avg"] = df["neg_count_1d_20d_avg"]
    df["neg_ratio_20d_avg"] = df["neg_ratio_1d_20d_avg"]
    df["sentiment_mean_20d_avg"] = df["sentiment_mean_1d_20d_avg"]

    df["abnormal_news_1d"] = df["news_count_1d"] / (df["news_count_20d_avg"] + 1)
    df["abnormal_news_3d"] = df["news_count_3d"] / (df["news_count_20d_avg"] + 1)

    df["abnormal_neg_count_1d"] = df["neg_count_1d"] / (df["neg_count_20d_avg"] + 1e-6)
    df["abnormal_neg_ratio_1d"] = df["neg_ratio_1d"] / (df["neg_ratio_20d_avg"] + 1e-6)

    df["abnormal_sentiment_drop_1d"] = (
        df["sentiment_mean_20d_avg"] - df["sentiment_mean_1d"]
    )

    for col in [
        "downgrade_news_count_1d",
        "guidance_cut_news_count_1d",
        "lawsuit_news_count_1d",
        "investigation_news_count_1d",
        "negative_event_count_1d",
    ]:
        base_col = f"{col}_20d_avg"
        df[f"abnormal_{col}"] = df[col] / (df[base_col] + 1e-6)

    for c in [
        "abnormal_news_1d", "abnormal_news_3d",
        "abnormal_neg_count_1d", "abnormal_neg_ratio_1d",
        "abnormal_downgrade_news_count_1d",
        "abnormal_guidance_cut_news_count_1d",
        "abnormal_lawsuit_news_count_1d",
        "abnormal_investigation_news_count_1d",
        "abnormal_negative_event_count_1d",
    ]:
        if c in df.columns:
            df[c] = df[c].clip(0, 10)

    df["sentiment_delta_1d"] = g["sentiment_mean_1d"].transform(lambda x: x.diff(1))
    df["sentiment_delta_3d"] = g["sentiment_mean_1d"].transform(lambda x: x.diff(3))

    df["neg_ratio_delta_1d"] = g["neg_ratio_1d"].transform(lambda x: x.diff(1))
    df["neg_ratio_delta_3d"] = g["neg_ratio_1d"].transform(lambda x: x.diff(3))

    df["news_count_delta_1d"] = g["news_count_1d"].transform(lambda x: x.diff(1))
    df["negative_event_count_delta_1d"] = g["negative_event_count_1d"].transform(lambda x: x.diff(1))
    df["downgrade_count_delta_1d"] = g["downgrade_news_count_1d"].transform(lambda x: x.diff(1))

    for c in [
        "sentiment_delta_1d", "sentiment_delta_3d",
        "neg_ratio_delta_1d", "neg_ratio_delta_3d",
        "news_count_delta_1d", "negative_event_count_delta_1d",
        "downgrade_count_delta_1d",
    ]:
        df[c] = df[c].fillna(0)

    df["news_spike_3d"] = df["news_count_1d"] / (
        g["news_count_1d"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) + 1
    )
    df["news_spike_7d"] = df["news_count_1d"] / (
        g["news_count_1d"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()) + 1
    )

    df["neg_spike_3d"] = df["neg_count_1d"] / (
        g["neg_count_1d"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) + 1
    )
    df["neg_spike_7d"] = df["neg_count_1d"] / (
        g["neg_count_1d"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()) + 1
    )

    event_spike_cols = [
        "downgrade_news_count_1d",
        "guidance_cut_news_count_1d",
        "lawsuit_news_count_1d",
        "investigation_news_count_1d",
        "negative_event_count_1d",
    ]

    for col in event_spike_cols:
        base_name = col.replace("_1d", "")

        df[f"{base_name}_spike_3d"] = df[col] / (
            g[col].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) + 1
        )
        df[f"{base_name}_spike_7d"] = df[col] / (
            g[col].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()) + 1
        )

    for c in [col for col in df.columns if "spike" in col]:
        df[c] = df[c].clip(0, 10)

    df["neg_day_flag"] = (df["neg_ratio_1d"] >= 0.5).astype(int)
    df["very_neg_day_flag"] = (df["very_neg_ratio_1d"] >= 0.2).astype(int)

    df["neg_streak_3d"] = g["neg_day_flag"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df["neg_streak_5d"] = g["neg_day_flag"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    df["negative_event_day_flag"] = (df["negative_event_ratio_1d"] > 0).astype(int)
    df["negative_event_streak_3d"] = g["negative_event_day_flag"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    df["negative_event_streak_5d"] = g["negative_event_day_flag"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    df["extreme_negative_flag"] = (df["neg_prob_mean_1d"] > 0.7).astype(int)

    df["panic_news"] = (
        (df["neg_ratio_1d"] > 0.6) &
        (df["news_count_1d"] > df["news_count_20d_avg"] * 1.5)
    ).astype(int)

    df["worst_sentiment_3d"] = g["sentiment_min_1d"].transform(
        lambda x: x.rolling(3, min_periods=1).min()
    )

    df["sentiment_shock"] = df["sentiment_mean_1d"] - df["sentiment_mean_20d_avg"]
    df["big_negative_shift"] = (df["sentiment_shock"] < -0.3).astype(int)

    df["panic_event_score"] = (
        2.0 * df["has_guidance_cut_news"] +
        2.0 * df["has_downgrade_news"] +
        2.0 * df["has_lawsuit_news"] +
        2.0 * df["has_investigation_news"] +
        2.0 * df["has_bankruptcy_news"] +
        1.5 * df["has_macro_risk_news"] +
        1.0 * df["has_analyst_news"] +
        1.0 * df["extreme_negative_article_ratio_1d"]
    )

    df["event_pressure_score"] = (
        df["panic_event_score"] *
        (1 + df["news_spike_3d"]) *
        (1 + df["neg_ratio_1d"])
    )

    df["negative_event_cluster"] = (
        df["has_downgrade_news"] +
        df["has_guidance_cut_news"] +
        df["has_lawsuit_news"] +
        df["has_investigation_news"] +
        df["has_bankruptcy_news"]
    )

    df["event_shock_score"] = (
        df["downgrade_news_count_spike_3d"] +
        df["guidance_cut_news_count_spike_3d"] +
        df["lawsuit_news_count_spike_3d"] +
        df["investigation_news_count_spike_3d"]
    )

    df["positive_event_but_negative_tone"] = (
        (
            df["has_product_news"] +
            df["has_upgrade_news"] +
            df["has_guidance_raise_news"]
        ) > 0
    ).astype(int) * (
        (df["sentiment_mean_1d"] < 0) |
        (df["neg_ratio_1d"] > 0.5) |
        (df["positive_event_negative_tone_ratio_1d"] > 0.3)
    ).astype(int)

    df["explicit_panic_news"] = (
        (df["panic_news"] == 1) &
        (df["negative_event_cluster"] > 0)
    ).astype(int)

    df["article_severity_score"] = (
        1.0 * df["strong_negative_article_ratio_1d"] +
        2.0 * df["extreme_negative_article_ratio_1d"] +
        1.5 * df["very_neg_ratio_1d"]
    )

    df["event_severity_score"] = (
        df["article_severity_score"] * (1 + df["negative_event_cluster"])
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = df.copy()

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def validate_news_features(df: pd.DataFrame) -> None:
    required = [
        "has_news",
        "news_count_1d",
        "news_count_3d",
        "news_count_7d",
        "news_count_log",
        "news_spike_7d",
        "sentiment_mean_1d",
        "sentiment_min_1d",
        "sentiment_max_1d",
        "sentiment_std_1d",
        "sentiment_delta_1d",
        "sentiment_delta_3d",
        "neg_ratio_1d",
        "neg_ratio_3d",
        "panic_news",
        "panic_event_score",
        "event_pressure_score",
        "negative_event_cluster",
        "event_shock_score",
        "explicit_panic_news",
        "article_severity_score",
        "event_severity_score",
        "has_earnings_news",
        "has_downgrade_news",
        "has_guidance_cut_news",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing news features: {missing}")

    if df[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate date/ticker rows found in news features.")