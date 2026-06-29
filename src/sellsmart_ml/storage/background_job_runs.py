from __future__ import annotations

import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Optional

from sellsmart_ml.storage.client import get_supabase
from sellsmart_ml.storage.supabase_predictions import clean_json_value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BackgroundJobRun:
    """Small helper for recording cron/job health in Supabase.

    The helper is deliberately non-fatal: if the table has not been created yet,
    cron jobs still run and print a warning instead of crashing.
    """

    def __init__(self, job_name: str, tickers_total: int = 0) -> None:
        self.job_name = job_name
        self.tickers_total = tickers_total
        self.id = str(uuid.uuid4())
        self._started_perf = perf_counter()
        self._enabled = True

    def mark_previous_running_as_failed(self) -> None:
        """Close stale rows left behind by an interrupted previous run.

        Render may show a cron execution as finished even if the process stopped
        before the final Supabase status update was written. Marking previous
        rows as failed keeps the monitoring table honest and prevents a stale
        "running" row from staying visible forever.
        """
        payload = clean_json_value(
            {
                "status": "failed",
                "completed_at": utc_now_iso(),
                "error_message": "Job was left running by a previous process",
            }
        )
        try:
            (
                get_supabase()
                .table("background_job_runs")
                .update(payload)
                .eq("job_name", self.job_name)
                .eq("status", "running")
                .execute()
            )
        except Exception as exc:
            print(f"[job-runs] Could not close stale running rows for {self.job_name}: {exc}")

    def start(self, details: Optional[dict[str, Any]] = None) -> None:
        self.mark_previous_running_as_failed()

        payload = clean_json_value(
            {
                "id": self.id,
                "job_name": self.job_name,
                "status": "running",
                "started_at": utc_now_iso(),
                "tickers_total": self.tickers_total,
                "tickers_succeeded": 0,
                "tickers_failed": 0,
                "details": details or {},
            }
        )
        try:
            get_supabase().table("background_job_runs").insert(payload).execute()
        except Exception as exc:
            self._enabled = False
            print(f"[job-runs] Could not create job run row for {self.job_name}: {exc}")

    def complete(
        self,
        *,
        tickers_succeeded: int,
        tickers_failed: int,
        details: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        status = "completed" if tickers_failed == 0 and not error_message else "failed"
        payload = clean_json_value(
            {
                "status": status,
                "completed_at": utc_now_iso(),
                "duration_ms": int((perf_counter() - self._started_perf) * 1000),
                "tickers_succeeded": tickers_succeeded,
                "tickers_failed": tickers_failed,
                "error_message": error_message,
                "details": details or {},
            }
        )
        if not self._enabled:
            return
        try:
            (
                get_supabase()
                .table("background_job_runs")
                .update(payload)
                .eq("id", self.id)
                .execute()
            )
        except Exception as exc:
            print(f"[job-runs] Could not update job run row for {self.job_name}: {exc}")
