from __future__ import annotations

from sellsmart_ml.inference.predict_live_risk import predict_ticker_risk
from sellsmart_ml.storage.supabase_prediction_jobs import (
    get_prediction_job,
    mark_prediction_job_completed,
    mark_prediction_job_failed,
    mark_prediction_job_processing,
    update_prediction_job,
)
from sellsmart_ml.storage.supabase_predictions import (
    get_latest_prediction,
    save_latest_prediction,
)


def process_prediction_job(job_id: str) -> None:
    """Process one queued prediction job.

    This can run as a FastAPI BackgroundTask now. Later, the same function can be
    called by a separate worker process if traffic grows.
    """
    job = get_prediction_job(job_id)
    if not job:
        print(f"[prediction-worker] Job not found: {job_id}")
        return

    ticker = str(job.get("ticker") or "").upper().strip()

    if not ticker:
        mark_prediction_job_failed(job_id, "Missing ticker.")
        return

    try:
        mark_prediction_job_processing(
            job_id,
            message="Checking prediction cache...",
            progress=15,
        )

        cached = get_latest_prediction(ticker)
        if cached is not None:
            mark_prediction_job_completed(job_id, cached)
            return

        update_prediction_job(
            job_id,
            message="Generating prediction from market data and news...",
            progress=35,
        )

        prediction = predict_ticker_risk(ticker)

        update_prediction_job(
            job_id,
            message="Saving generated prediction...",
            progress=85,
        )

        if prediction is not None:
            save_latest_prediction(prediction)

        mark_prediction_job_completed(job_id, prediction or {})

    except Exception as exc:
        print(f"[prediction-worker] Failed job {job_id} for {ticker}: {exc}")
        mark_prediction_job_failed(job_id, str(exc))
