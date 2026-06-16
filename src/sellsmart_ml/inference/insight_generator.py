from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HORIZON = "5d"


@dataclass
class Driver:
    feature: str
    label: str
    direction: str
    impact: str
    value: float | int | str
    message: str


def _get(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = row.get(col, default)
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def get_risk_category(risk_score: int) -> str:
    if risk_score >= 75:
        return "high"
    if risk_score >= 55:
        return "medium"
    if risk_score >= 35:
        return "elevated"
    return "low"


def get_category_label(category: str) -> str:
    return {
        "high": "High Risk",
        "medium": "Medium Risk",
        "elevated": "Elevated Risk",
        "low": "Low Risk",
    }[category]


def get_recommended_action(category: str) -> str:
    return {
        "high": "reduce",
        "medium": "wait",
        "elevated": "monitor",
        "low": "hold",
    }[category]


def get_action_label(action: str) -> str:
    return {
        "reduce": "Reduce Exposure",
        "wait": "Wait / Avoid New Entry",
        "monitor": "Monitor Closely",
        "hold": "Hold - No Panic Signal",
    }[action]


def get_news_status(row: pd.Series) -> str:
    explicit_status = str(row.get("news_status", "")).lower()

    if explicit_status in {"live", "cached", "fallback", "synthetic", "none"}:
        return explicit_status

    source = str(row.get("source", "")).lower()

    if source == "fallback":
        return "fallback"

    has_news = _get(row, "has_news")
    news_count_1d = _get(row, "news_count_1d")
    news_count_3d = _get(row, "news_count_3d")
    news_count_7d = _get(row, "news_count_7d")

    if has_news > 0 or news_count_1d > 0 or news_count_3d > 0 or news_count_7d > 0:
        return "live"

    return "none"


def get_market_regime(row: pd.Series, category: str) -> str:
    ret_5 = _get(row, "ret_5")
    drawdown_20 = _get(row, "drawdown_20")
    vol_ratio_20 = _get(row, "vol_ratio_20")
    rsi_14 = _get(row, "rsi_14")
    vix = _get(row, "VIX")

    strong_selloff = ret_5 <= -0.12 or drawdown_20 <= -0.20
    moderate_selloff = ret_5 <= -0.05 or drawdown_20 <= -0.08
    oversold = 0 < rsi_14 <= 30
    high_volume = vol_ratio_20 >= 1.5
    calm_market = 0 < vix < 20

    if category == "low" and strong_selloff and oversold:
        return "Post-selloff stabilization"

    if category == "low" and strong_selloff and high_volume:
        return "Stress present, panic not escalating"

    if category == "low" and strong_selloff:
        return "Pullback, but no panic escalation"

    if category == "low" and moderate_selloff:
        return "Controlled pullback"

    if category == "low" and calm_market:
        return "Calm market behavior"

    if category == "elevated":
        return "Early warning signs"

    if category == "medium":
        return "Stress building"

    if category == "high":
        return "Active panic-risk setup"

    return "Mixed market conditions"


def get_confidence(
    probability: float,
    threshold: float,
    row: pd.Series,
) -> str:
    distance = abs(probability - threshold)

    has_price_context = _get(row, "vol_20") > 0
    news_status = get_news_status(row)
    has_live_news = news_status == "live"

    if distance >= 0.25 and has_price_context and has_live_news:
        return "high"

    if distance >= 0.25 and has_price_context:
        return "medium"

    if distance >= 0.12 and has_price_context:
        return "medium"

    return "low"


def get_confidence_explanation(confidence: str) -> str:
    return {
        "high": "The model signal is relatively clear because the probability is far from the decision threshold.",
        "medium": "The model signal is usable, but some indicators may be mixed.",
        "low": "Signals are mixed or close to the decision threshold, so this should be interpreted cautiously.",
    }[confidence]


def get_signal_conflict(row: pd.Series) -> str | None:
    ret_3 = _get(row, "ret_3")
    ret_5 = _get(row, "ret_5")
    neg_ratio_3d = _get(row, "neg_ratio_3d")
    neg_count_1d = _get(row, "neg_count_1d")
    vol_ratio_20 = _get(row, "vol_ratio_20")
    rsi_14 = _get(row, "rsi_14")

    if neg_ratio_3d >= 0.45 and ret_5 > 0.03:
        return (
            "News sentiment is negative, but price momentum remains positive. "
            "This creates a mixed signal rather than a one-sided panic setup."
        )

    if neg_count_1d >= 5 and neg_ratio_3d < 0.35:
        return (
            "There has been some recent negative coverage, but the latest live news tone "
            "does not appear strongly negative."
        )

    if ret_3 >= 0.06 and vol_ratio_20 < 1.1:
        return (
            "The stock has moved up quickly, but trading volume is not showing obvious panic behavior."
        )

    if rsi_14 <= 35 and ret_5 > 0:
        return (
            "Technical stress is present, but recent price action has started to recover."
        )

    return None


def get_action_explanation(category: str, row: pd.Series | None = None) -> str:
    if category == "high":
        return (
            "The model sees signs that downside risk may be elevated. "
            "This may be a moment to reduce exposure, avoid adding more, "
            "or wait for conditions to stabilize."
        )

    if category == "medium":
        return (
            "Risk is meaningful, but not extreme. A calm wait-and-review approach may be better "
            "than reacting emotionally to short-term market pressure."
        )

    if category == "elevated":
        return (
            "Some warning signs are present, but the model does not yet see a strong panic-risk setup. "
            "This may be a situation to monitor rather than react to immediately."
        )

    if row is not None:
        ret_5 = _get(row, "ret_5")
        drawdown_20 = _get(row, "drawdown_20")
        rsi_14 = _get(row, "rsi_14")
        vol_ratio_20 = _get(row, "vol_ratio_20")

        strong_selloff = ret_5 <= -0.12 or drawdown_20 <= -0.20
        moderate_selloff = ret_5 <= -0.05 or drawdown_20 <= -0.08
        oversold = 0 < rsi_14 <= 30

        if strong_selloff and oversold:
            return (
                "The stock has already experienced a sharp selloff, which can feel alarming. "
                "However, the model does not currently detect signs that panic-selling pressure "
                "is escalating further."
            )

        if strong_selloff and vol_ratio_20 < 1.2:
            return (
                "The stock is still below recent highs, but trading activity does not show obvious panic behavior. "
                "The model sees the current setup as more controlled than distressed."
            )

        if moderate_selloff:
            return (
                "The stock has pulled back, but current market behavior does not suggest strong panic-selling pressure. "
                "The model sees this as a situation to stay calm and review rather than react emotionally."
            )

    return (
        "Current downside risk appears contained. The model does not see strong evidence of panic-selling pressure."
    )


def build_summary(
    ticker: str,
    category: str,
    risk_score: int,
    probability: float,
    row: pd.Series | None = None,
) -> str:
    if category == "high":
        return (
            f"{ticker} shows elevated short-term downside risk. "
            f"The model estimates a {probability:.1%} probability of a significant 5-day drawdown, "
            f"with a risk score of {risk_score}/100. The signal suggests that rapid price movement, "
            f"volatility, and market/news pressure may be combining into a panic-risk setup."
        )

    if category == "medium":
        return (
            f"{ticker} shows moderate short-term downside risk. "
            f"The model estimates a {probability:.1%} probability of a significant 5-day drawdown. "
            f"The setup deserves attention, but the signal is not yet extreme."
        )

    if category == "elevated":
        return (
            f"{ticker} has some early warning signs, but the model does not yet classify the risk as high. "
            f"The current risk score is {risk_score}/100."
        )

    if row is not None:
        ret_5 = _get(row, "ret_5")
        drawdown_20 = _get(row, "drawdown_20")
        rsi_14 = _get(row, "rsi_14")

        strong_selloff = ret_5 <= -0.12 or drawdown_20 <= -0.20
        oversold = 0 < rsi_14 <= 30

        if strong_selloff and oversold:
            return (
                f"{ticker} has already gone through a sharp selloff, but the model currently sees low "
                f"short-term panic risk. It estimates a {probability:.1%} probability of a significant "
                f"5-day drawdown. This suggests the situation may be stressed, but not necessarily escalating."
            )

        if strong_selloff:
            return (
                f"{ticker} is still trading well below recent highs, but the model currently sees low "
                f"short-term panic risk. It estimates a {probability:.1%} probability of a significant "
                f"5-day drawdown."
            )

    return (
        f"{ticker} shows low short-term panic risk. "
        f"The model estimates a {probability:.1%} probability of a significant 5-day drawdown. "
        f"This does not mean the stock is risk-free, but current market behavior does not suggest "
        f"strong panic-selling pressure."
    )


def generate_price_drivers(row: pd.Series) -> list[Driver]:
    drivers: list[Driver] = []

    ret_2 = _get(row, "ret_2")
    ret_3 = _get(row, "ret_3")
    ret_5 = _get(row, "ret_5")
    vol_ratio_20 = _get(row, "vol_ratio_20")
    drawdown_20 = _get(row, "drawdown_20")
    rsi_14 = _get(row, "rsi_14")
    vix = _get(row, "VIX")
    vix_spike = _get(row, "VIX_spike_5d")
    market_vol = _get(row, "market_volatility_20d")
    dist_ma_20 = _get(row, "dist_ma_20")

    if ret_3 >= 0.06:
        drivers.append(Driver(
            feature="ret_3",
            label="Sharp recent rally",
            direction="negative",
            impact="high",
            value=round(ret_3, 4),
            message=(
                "The stock has risen very quickly over the last few sessions. "
                "Fast rallies can increase the chance of a short-term pullback."
            ),
        ))
    elif ret_2 >= 0.06:
        drivers.append(Driver(
            feature="ret_2",
            label="Fast two-day move",
            direction="negative",
            impact="high",
            value=round(ret_2, 4),
            message=(
                "The stock has made a fast short-term move upward, which can increase "
                "the probability of a near-term pullback."
            ),
        ))

    if ret_5 <= -0.05:
        drivers.append(Driver(
            feature="ret_5",
            label="Recent price weakness",
            direction="negative",
            impact="high",
            value=round(ret_5, 4),
            message=(
                "The stock has fallen sharply over the last few trading days. "
                "This can feel alarming for investors and may increase emotional selling risk."
            ),
        ))
    elif ret_3 <= -0.03:
        drivers.append(Driver(
            feature="ret_3",
            label="Short-term selloff",
            direction="negative",
            impact="medium",
            value=round(ret_3, 4),
            message="The stock is under short-term selling pressure.",
        ))

    if drawdown_20 <= -0.08:
        drivers.append(Driver(
            feature="drawdown_20",
            label="Meaningful drawdown",
            direction="negative",
            impact="high",
            value=round(drawdown_20, 4),
            message=(
                "The stock is trading well below its recent high. "
                "This can make investors nervous, even when panic pressure is not accelerating."
            ),
        ))

    if vol_ratio_20 >= 1.5:
        drivers.append(Driver(
            feature="vol_ratio_20",
            label="Unusual trading volume",
            direction="negative",
            impact="medium",
            value=round(vol_ratio_20, 2),
            message="Trading volume is elevated versus the recent average, suggesting a stronger market reaction.",
        ))

    if rsi_14 <= 35:
        drivers.append(Driver(
            feature="rsi_14",
            label="Oversold technical condition",
            direction="mixed",
            impact="medium",
            value=round(rsi_14, 1),
            message=(
                "The stock looks technically oversold. This can signal market stress, "
                "but it may also mean some selling pressure has already played out."
            ),
        ))

    if dist_ma_20 <= -0.06:
        drivers.append(Driver(
            feature="dist_ma_20",
            label="Below medium-term trend",
            direction="negative",
            impact="medium",
            value=round(dist_ma_20, 4),
            message="Price is meaningfully below its 20-day moving average, indicating trend deterioration.",
        ))

    if vix >= 25:
        drivers.append(Driver(
            feature="VIX",
            label="High market fear",
            direction="negative",
            impact="high",
            value=round(vix, 2),
            message="The broader market fear index is elevated, which can amplify downside moves.",
        ))
    elif vix >= 20:
        drivers.append(Driver(
            feature="VIX",
            label="Elevated market stress",
            direction="negative",
            impact="medium",
            value=round(vix, 2),
            message="Market stress is above calm levels, increasing the chance of risk-off behavior.",
        ))

    if vix_spike >= 0.15:
        drivers.append(Driver(
            feature="VIX_spike_5d",
            label="Fear spike",
            direction="negative",
            impact="medium",
            value=round(vix_spike, 4),
            message="Market fear has risen quickly compared with its recent average.",
        ))

    if market_vol >= 0.018:
        drivers.append(Driver(
            feature="market_volatility_20d",
            label="Volatile market regime",
            direction="negative",
            impact="medium",
            value=round(market_vol, 4),
            message="The broader market is in a more volatile regime, which can make individual stock signals less stable.",
        ))

    return drivers


def generate_news_drivers(row: pd.Series) -> list[Driver]:
    drivers: list[Driver] = []

    news_count_1d = _get(row, "news_count_1d")
    news_count_3d = _get(row, "news_count_3d")
    news_count_7d = _get(row, "news_count_7d")

    neg_count_1d = _get(row, "neg_count_1d")
    very_neg_count_1d = _get(row, "very_neg_count_1d")

    neg_ratio_1d = _get(row, "neg_ratio_1d")
    neg_ratio_3d = _get(row, "neg_ratio_3d")

    sentiment_mean_1d = _get(row, "sentiment_mean_1d")
    sentiment_min_1d = _get(row, "sentiment_min_1d")

    news_spike_7d = _get(row, "news_spike_7d")
    panic_news = _get(row, "panic_news")

    event_pressure_score = _get(row, "event_pressure_score")
    negative_event_cluster = _get(row, "negative_event_cluster")
    article_severity_score = _get(row, "article_severity_score")

    earnings_news_count3d = _get(row, "earnings_news_count3d")
    guidance_cut_news_count3d = _get(row, "guidance_cut_news_count3d")

    # 1. Strongest negative sentiment signal only
    if very_neg_count_1d >= 3:
        drivers.append(Driver(
            feature="very_neg_count_1d",
            label="Very negative news cluster",
            direction="negative",
            impact="high",
            value=int(very_neg_count_1d),
            message="Several recent articles have strongly negative sentiment, which may increase panic-risk pressure.",
        ))
    elif news_count_1d > 0 and neg_ratio_1d >= 0.6:
        drivers.append(Driver(
            feature="neg_ratio_1d",
            label="Negative news tone",
            direction="negative",
            impact="high",
            value=round(neg_ratio_1d, 3),
            message="A high share of recent news coverage is negative, which can increase emotional pressure on investors.",
        ))
    elif sentiment_min_1d <= -0.90:
        drivers.append(Driver(
            feature="sentiment_min_1d",
            label="Severe negative article tone",
            direction="negative",
            impact="medium",
            value=round(sentiment_min_1d, 3),
            message="At least one recent article has extremely negative sentiment.",
        ))
    elif sentiment_mean_1d <= -0.35:
        drivers.append(Driver(
            feature="sentiment_mean_1d",
            label="Weak news sentiment",
            direction="negative",
            impact="medium",
            value=round(sentiment_mean_1d, 3),
            message="Average recent news sentiment is clearly negative.",
        ))
    elif news_count_7d > 0 and neg_ratio_3d >= 0.45:
        drivers.append(Driver(
            feature="neg_ratio_3d",
            label="Negative news trend",
            direction="negative",
            impact="medium",
            value=round(neg_ratio_3d, 3),
            message="News tone has been leaning negative over the last few days.",
        ))

    # 2. Negative coverage volume
    if neg_count_1d >= 5:
        drivers.append(Driver(
            feature="neg_count_1d",
            label="Elevated recent negative coverage",
            direction="negative",
            impact="medium",
            value=int(neg_count_1d),
            message="Several negative articles appeared recently, which may have influenced investor sentiment.",
        ))
    elif neg_count_1d >= 2:
        drivers.append(Driver(
            feature="neg_count_1d",
            label="Some recent negative coverage",
            direction="negative",
            impact="low",
            value=int(neg_count_1d),
            message="A few recent articles carried negative sentiment, but the negative coverage is not unusually heavy.",
        ))

    # 3. General news activity — this was missing
    if news_count_1d >= 30 or news_spike_7d >= 2.0:
        value = int(news_count_1d) if news_count_1d >= 30 else round(news_spike_7d, 2)
        drivers.append(Driver(
            feature="news_activity",
            label="Elevated news activity",
            direction="negative",
            impact="medium",
            value=value,
            message="News coverage is significantly higher than normal, which can amplify emotional reactions.",
        ))
    elif news_count_1d >= 5 or news_count_3d >= 10:
        drivers.append(Driver(
            feature="news_activity",
            label="Active news coverage",
            direction="mixed",
            impact="low",
            value=int(news_count_1d or news_count_3d),
            message="The stock is receiving active news coverage, which may increase investor attention even if tone is not strongly negative.",
        ))

    # 4. Event-specific signals
    if guidance_cut_news_count3d >= 1:
        drivers.append(Driver(
            feature="guidance_cut_news_count3d",
            label="Guidance-cut signal detected",
            direction="negative",
            impact="high",
            value=int(guidance_cut_news_count3d),
            message="Recent coverage contains guidance-cut language, which can increase downside pressure.",
        ))

    if panic_news >= 1:
        drivers.append(Driver(
            feature="panic_news",
            label="Panic-news pattern",
            direction="negative",
            impact="high",
            value=int(panic_news),
            message="The model detected a combination of negative tone and abnormal news volume.",
        ))

    if negative_event_cluster >= 2:
        drivers.append(Driver(
            feature="negative_event_cluster",
            label="Cluster of negative events",
            direction="negative",
            impact="high",
            value=int(negative_event_cluster),
            message="Multiple negative event categories appear in recent coverage.",
        ))
    elif negative_event_cluster == 1:
        drivers.append(Driver(
            feature="negative_event_cluster",
            label="Negative event detected",
            direction="negative",
            impact="medium",
            value=int(negative_event_cluster),
            message="Recent news contains at least one negative event signal such as downgrade, lawsuit, investigation, or guidance cut.",
        ))

    if event_pressure_score >= 8:
        drivers.append(Driver(
            feature="event_pressure_score",
            label="High event pressure",
            direction="negative",
            impact="high",
            value=round(event_pressure_score, 2),
            message="Event-related news pressure is elevated and may be contributing to downside risk.",
        ))

    if earnings_news_count3d >= 5:
        drivers.append(Driver(
            feature="earnings_news_count3d",
            label="Earnings-related attention",
            direction="mixed",
            impact="medium",
            value=int(earnings_news_count3d),
            message="There is elevated earnings-related coverage, which can increase short-term volatility and investor attention.",
        ))

    already_has_severe_tone = any(
        d.feature in {"very_neg_count_1d", "sentiment_min_1d", "sentiment_mean_1d"}
        for d in drivers
    )

    if article_severity_score >= 1.5 and not already_has_severe_tone:
        drivers.append(Driver(
            feature="article_severity_score",
            label="Severe article tone",
            direction="negative",
            impact="medium",
            value=round(article_severity_score, 2),
            message="Recent articles contain unusually severe negative tone.",
        ))

    return drivers

def generate_supportive_signals(row: pd.Series) -> list[Driver]:
    drivers: list[Driver] = []

    ret_3 = _get(row, "ret_3")
    ret_5 = _get(row, "ret_5")
    vol_ratio_20 = _get(row, "vol_ratio_20")
    rsi_14 = _get(row, "rsi_14")
    neg_count_1d = _get(row, "neg_count_1d")
    neg_ratio_3d = _get(row, "neg_ratio_3d")
    vix = _get(row, "VIX")

    # Avoid direct contradiction with "Sharp recent rally".
    if 0.01 <= ret_5 < 0.05 and ret_3 < 0.06:
        drivers.append(Driver(
            feature="ret_5",
            label="Constructive price momentum",
            direction="positive",
            impact="low",
            value=round(ret_5, 4),
            message=(
                "Price momentum is positive, but not extreme enough to suggest an overheated short-term move."
            ),
        ))

    # If there is a sharp short-term rally, phrase the supportive signal differently.
    elif ret_5 > 0.01 and ret_3 >= 0.06:
        drivers.append(Driver(
            feature="ret_5",
            label="Broader trend remains positive",
            direction="positive",
            impact="low",
            value=round(ret_5, 4),
            message=(
                "Despite the fast recent rally, the broader price trend remains constructive."
            ),
        ))

    if 0.3 <= vol_ratio_20 < 1.1:
        drivers.append(Driver(
            feature="vol_ratio_20",
            label="Normal trading volume",
            direction="positive",
            impact="low",
            value=round(vol_ratio_20, 2),
            message=(
                "Volume is close to normal, suggesting no obvious panic-volume spike."
            ),
        ))

    if 40 <= rsi_14 <= 65:
        drivers.append(Driver(
            feature="rsi_14",
            label="Balanced technical condition",
            direction="positive",
            impact="low",
            value=round(rsi_14, 1),
            message=(
                "RSI is in a balanced range, without clear oversold stress."
            ),
        ))

    # Avoid saying "no negative news pressure" when we also show many negative articles.
    if get_news_status(row) == "live" and 0 <= neg_ratio_3d < 0.35:
        if neg_count_1d >= 5:
            drivers.append(Driver(
                feature="neg_ratio_3d",
                label="Current news tone is stabilizing",
                direction="positive",
                impact="low",
                value=round(neg_ratio_3d, 3),
                message=(
                    "There was some recent negative coverage, but the latest live news tone does not appear strongly negative."
                ),
            ))
        else:
            drivers.append(Driver(
                feature="neg_ratio_3d",
                label="No strong negative news pressure",
                direction="positive",
                impact="low",
                value=round(neg_ratio_3d, 3),
                message=(
                    "Recent live news tone does not look strongly negative."
                ),
            ))

    if 0 < vix < 20:
        drivers.append(Driver(
            feature="VIX",
            label="Calmer market backdrop",
            direction="positive",
            impact="low",
            value=round(vix, 2),
            message=(
                "The broader market fear index is not in a high-stress zone."
            ),
        ))

    return drivers


def rank_drivers(drivers: list[Driver], max_drivers: int = 5) -> list[Driver]:
    impact_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    direction_rank = {
        "negative": 3,
        "mixed": 2,
        "positive": 1,
    }

    return sorted(
        drivers,
        key=lambda d: (
            impact_rank.get(d.impact, 0),
            direction_rank.get(d.direction, 0),
        ),
        reverse=True,
    )[:max_drivers]


def filter_low_risk_stress_signals(drivers: list[Driver]) -> list[Driver]:
    """
    For very low-risk stocks, avoid showing too many scary yellow/red news boxes.
    Keep only the clearest warnings.
    """
    priority_features = {
        "very_neg_count_1d",
        "neg_ratio_1d",
        "guidance_cut_news_count3d",
        "panic_news",
        "negative_event_cluster",
        "event_pressure_score",
        "ret_5",
        "drawdown_20",
        "vol_ratio_20",
    }

    filtered = [
        d for d in drivers
        if d.impact == "high" or d.feature in priority_features
    ]

    return rank_drivers(filtered, max_drivers=3)


def filter_supportive_signals(
    supportive_signals: list[Driver],
    negative_drivers: list[Driver],
) -> list[Driver]:
    negative_features = {d.feature for d in negative_drivers}
    negative_labels = {d.label for d in negative_drivers}

    filtered: list[Driver] = []

    for signal in supportive_signals:
        # Prevent direct feature-level contradiction.
        if signal.feature in negative_features and signal.feature != "ret_5":
            continue

        # Prevent direct wording contradiction with sharp rally.
        if (
            signal.label == "Constructive price momentum"
            and "Sharp recent rally" in negative_labels
        ):
            continue

        filtered.append(signal)

    return filtered[:3]


def driver_to_dict(driver: Driver) -> dict[str, Any]:
    return {
        "feature": driver.feature,
        "label": driver.label,
        "direction": driver.direction,
        "impact": driver.impact,
        "value": driver.value,
        "message": driver.message,
    }


def generate_insight(
    ticker: str,
    probability: float,
    threshold: float,
    latest_row: pd.Series,
) -> dict[str, Any]:

    risk_score = int(round(probability * 100))
    category = get_risk_category(risk_score)
    category_label = get_category_label(category)

    action = get_recommended_action(category)
    action_label = get_action_label(action)

    confidence = get_confidence(
        probability=probability,
        threshold=threshold,
        row=latest_row,
    )

    news_status = get_news_status(latest_row)
    market_regime = get_market_regime(latest_row, category)

    negative_drivers = (
        generate_price_drivers(latest_row)
        + generate_news_drivers(latest_row)
    )

    supportive_signals = generate_supportive_signals(latest_row)
    supportive_signals = filter_supportive_signals(
        supportive_signals=supportive_signals,
        negative_drivers=negative_drivers,
    )

    ranked_negative_signals = rank_drivers(
        negative_drivers,
        max_drivers=5,
    )

    if category == "low":
        top_drivers = []
        stress_signals = filter_low_risk_stress_signals(ranked_negative_signals)
    else:
        top_drivers = ranked_negative_signals
        stress_signals = []

    if category == "high" and len(top_drivers) == 0:
        top_drivers.append(Driver(
            feature="model_signal",
            label="High model risk signal",
            direction="negative",
            impact="high",
            value=round(probability, 4),
            message=(
                "The model detects elevated downside risk, but the explanation layer "
                "did not identify a clear individual driver."
            ),
        ))

    if category == "high" and len(top_drivers) < 4:
        top_drivers.append(Driver(
            feature="model_consensus",
            label="Strong model consensus",
            direction="negative",
            impact="high",
            value=round(probability, 4),
            message=(
                "Multiple independent indicators are pointing "
                "toward elevated downside risk."
            ),
        ))

    signal_conflict = get_signal_conflict(latest_row)

    return {
        "ticker": ticker,
        "date": str(latest_row.get("date")),
        "horizon": HORIZON,

        "current_price": round(
            float(latest_row.get("close", 0)),
            2,
        ),

        "risk_score": risk_score,
        "probability_of_drop": round(probability, 4),

        "category": category,
        "category_label": category_label,

        "action": action,
        "action_label": action_label,

        "confidence": confidence,
        "confidence_explanation": get_confidence_explanation(confidence),

        "news_status": news_status,
        "market_regime": market_regime,
        "signal_conflict": signal_conflict,

        "summary": build_summary(
            ticker=ticker,
            category=category,
            risk_score=risk_score,
            probability=probability,
            row=latest_row,
        ),
        "action_explanation": get_action_explanation(category, latest_row),

        "drivers": [
            driver_to_dict(driver)
            for driver in top_drivers
        ],

        "stress_signals": [
            driver_to_dict(driver)
            for driver in stress_signals[:3]
        ],

        "supportive_signals": [
            driver_to_dict(driver)
            for driver in supportive_signals[:3]
        ],

        "disclaimer": (
            "This is a model-based risk signal, not financial advice. "
            "Use it as decision support together with your own judgment."
        ),
    }