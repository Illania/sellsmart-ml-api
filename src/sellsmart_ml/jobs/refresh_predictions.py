from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk
from sellsmart_ml.storage.supabase_predictions import (
    save_latest_prediction,
)


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
    "CRM",
    "ADBE",
    "INTC",
    "QCOM",
    "PYPL",
    "INSM",
]


def main() -> None:

    print("Starting predictions refresh...")

    for ticker in TICKERS:

        print("")
        print(f"Predicting {ticker}...")

        try:
            result = predict_ticker_risk(
                ticker=ticker,
                force_refresh_news=True,
                force_refresh_prices=True,
                force_refresh_market=False,
            )

            save_latest_prediction(result)

            print(
                f"Saved prediction for {ticker} | "
                f"risk_score={result.get('risk_score')} | "
                f"category={result.get('category')}"
            )

        except Exception as exc:
            print(f"ERROR {ticker}: {exc}")

    print("")
    print("Predictions refresh completed.")


if __name__ == "__main__":
    main()