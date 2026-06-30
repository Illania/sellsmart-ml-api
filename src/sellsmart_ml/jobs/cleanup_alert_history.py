from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from sellsmart_ml.storage.alert_history import cleanup_all_user_alert_history
from sellsmart_ml.storage.background_job_runs import BackgroundJobRun


def main() -> None:
    print("Starting alert history cleanup...")
    job_run = BackgroundJobRun("cleanup_alert_history")
    job_run.start()

    try:
        result = cleanup_all_user_alert_history()
        errors = result.get("errors") or []
        job_run.complete(
            tickers_succeeded=int(result.get("users_processed") or 0),
            tickers_failed=len(errors),
            details=result,
            error_message=f"{len(errors)} user(s) failed" if errors else None,
        )
        print(f"Alert history cleanup completed: {result}")
    except Exception as exc:
        job_run.complete(
            tickers_succeeded=0,
            tickers_failed=1,
            details={"fatal_error": str(exc)},
            error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()
