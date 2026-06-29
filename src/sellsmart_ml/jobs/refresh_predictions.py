from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk
from sellsmart_ml.storage.supabase_predictions import (
    save_latest_prediction,
)
from sellsmart_ml.storage.background_job_runs import BackgroundJobRun
from sellsmart_ml.storage.ticker_universe import get_background_refresh_tickers


def main() -> None:

    print("Starting predictions refresh...")

    tickers = get_background_refresh_tickers()
    job_run = BackgroundJobRun("refresh_predictions", tickers_total=len(tickers))
    job_run.start(details={"tickers": tickers})

    succeeded = 0
    failed = 0
    errors: list[dict[str, str]] = []

    try:
        for ticker in tickers:

            print("")
            print(f"Predicting {ticker}...")

            try:
                result = predict_ticker_risk(
                    ticker=ticker,
                    force_refresh_news=True,
                    force_refresh_prices=True,
                    # Daily cron should refresh broad market context too, otherwise
                    # SPY/QQQ/VIX regime features may lag behind fresh ticker prices.
                    force_refresh_market=True,
                )

                save_latest_prediction(result)
                succeeded += 1

                print(
                    f"Saved prediction for {ticker} | "
                    f"risk_score={result.get('risk_score')} | "
                    f"category={result.get('category')}"
                )

            except Exception as exc:
                failed += 1
                errors.append({"ticker": ticker, "error": str(exc)})
                print(f"ERROR {ticker}: {exc}")

        job_run.complete(
            tickers_succeeded=succeeded,
            tickers_failed=failed,
            details={"tickers": tickers, "errors": errors[:50]},
            error_message=f"{failed} ticker(s) failed" if failed else None,
        )

    except Exception as exc:
        job_run.complete(
            tickers_succeeded=succeeded,
            tickers_failed=failed + 1,
            details={"tickers": tickers, "errors": errors[:50], "fatal_error": str(exc)},
            error_message=str(exc),
        )
        raise

    print("")
    print("Predictions refresh completed.")


if __name__ == "__main__":
    main()