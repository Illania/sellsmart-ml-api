from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sellsmart_ml.storage.client import get_supabase

DEFAULT_ALERT_HISTORY_DAYS = 90


def _normalize_retention_days(value: Any) -> int | None:
    """Return retention days. None means keep forever."""
    if value is None:
        return None

    try:
        days = int(value)
    except (TypeError, ValueError):
        return DEFAULT_ALERT_HISTORY_DAYS

    if days <= 0:
        return None

    return days


def cleanup_user_alert_history(user_id: str, retention_days: int | None) -> int:
    """Delete acknowledged alerts older than the user's retention window."""
    if retention_days is None:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    supabase = get_supabase()

    response = (
        supabase.table("read_alerts")
        .delete()
        .eq("user_id", user_id)
        .lt("read_at", cutoff.isoformat())
        .execute()
    )

    deleted_rows = getattr(response, "data", None) or []
    return len(deleted_rows)


def cleanup_all_user_alert_history() -> dict[str, Any]:
    """Apply alert-history retention settings for all users.

    user_settings.alert_history_days stores the retention period in days.
    NULL means keep alert history forever.
    """
    supabase = get_supabase()

    response = (
        supabase.table("user_settings")
        .select("user_id, alert_history_days")
        .execute()
    )

    rows = getattr(response, "data", None) or []
    users_processed = 0
    users_skipped = 0
    deleted_total = 0
    errors: list[dict[str, str]] = []

    for row in rows:
        user_id = row.get("user_id")
        if not user_id:
            users_skipped += 1
            continue

        retention_days = _normalize_retention_days(row.get("alert_history_days"))

        if retention_days is None:
            users_skipped += 1
            continue

        try:
            deleted_total += cleanup_user_alert_history(user_id, retention_days)
            users_processed += 1
        except Exception as exc:  # pragma: no cover - job should continue for other users
            errors.append({"user_id": str(user_id), "error": str(exc)})

    return {
        "users_total": len(rows),
        "users_processed": users_processed,
        "users_skipped": users_skipped,
        "deleted_read_alerts": deleted_total,
        "errors": errors[:50],
    }
