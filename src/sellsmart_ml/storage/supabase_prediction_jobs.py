from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sellsmart_ml.storage.client import get_supabase
from sellsmart_ml.storage.supabase_predictions import clean_json_value


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
ACTIVE_STATUSES = {"pending", "processing"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_prediction_job(
    ticker: str,
    user_id: Optional[str] = None,
) -> dict:
    """Create a queued prediction job in Supabase.

    For MVP, the queue is a database table. A FastAPI BackgroundTask can pick up
    the job immediately, and the same table can later be processed by a separate
    worker/cron without changing the API contract.
    """
    supabase = get_supabase()
    now = utc_now_iso()
    job = clean_json_value(
        {
            "id": str(uuid.uuid4()),
            "ticker": ticker.upper(),
            "user_id": user_id,
            "status": "pending",
            "progress": 0,
            "message": "Prediction request queued.",
            "created_at": now,
            "updated_at": now,
        }
    )

    response = supabase.table("prediction_jobs").insert(job).execute()
    if response.data:
        return response.data[0]
    return job


def get_prediction_job(job_id: str) -> Optional[dict]:
    supabase = get_supabase()
    response = (
        supabase
        .table("prediction_jobs")
        .select("*")
        .eq("id", job_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def get_active_prediction_job(ticker: str, user_id: Optional[str] = None) -> Optional[dict]:
    """Return the newest active job for this ticker/user, if one exists.

    This prevents multiple browser refreshes from starting several expensive
    FinBERT/yfinance prediction runs for the same user ticker.
    """
    supabase = get_supabase()
    query = (
        supabase
        .table("prediction_jobs")
        .select("*")
        .eq("ticker", ticker.upper())
        .in_("status", list(ACTIVE_STATUSES))
        .order("created_at", desc=True)
        .limit(1)
    )

    if user_id:
        query = query.eq("user_id", user_id)

    response = query.execute()
    if not response.data:
        return None
    return response.data[0]


def update_prediction_job(job_id: str, **fields: Any) -> Optional[dict]:
    supabase = get_supabase()
    payload = clean_json_value({**fields, "updated_at": utc_now_iso()})
    response = (
        supabase
        .table("prediction_jobs")
        .update(payload)
        .eq("id", job_id)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def mark_prediction_job_processing(job_id: str, message: str, progress: int = 10) -> Optional[dict]:
    return update_prediction_job(
        job_id,
        status="processing",
        message=message,
        progress=progress,
        started_at=utc_now_iso(),
    )


def mark_prediction_job_completed(job_id: str, prediction: dict) -> Optional[dict]:
    return update_prediction_job(
        job_id,
        status="completed",
        message="Prediction ready.",
        progress=100,
        prediction_json=prediction,
        error_message=None,
        completed_at=utc_now_iso(),
    )


def mark_prediction_job_failed(job_id: str, error_message: str) -> Optional[dict]:
    return update_prediction_job(
        job_id,
        status="failed",
        message="Prediction failed.",
        progress=100,
        error_message=error_message,
        completed_at=utc_now_iso(),
    )


def serialize_prediction_job(job: dict) -> dict:
    """Return a frontend-friendly job shape."""
    return {
        "job_id": job.get("id"),
        "ticker": job.get("ticker"),
        "status": job.get("status"),
        "progress": job.get("progress") or 0,
        "message": job.get("message"),
        "prediction": job.get("prediction_json"),
        "error_message": job.get("error_message"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
    }
