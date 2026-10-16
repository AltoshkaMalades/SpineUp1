"""Pure calculation helpers for local posture session statistics."""

from datetime import date, datetime, timedelta
from typing import Any


def daily_session_stats(
    sessions: list[tuple[str, int, int]], days: int, today: date
) -> list[dict[str, Any]]:
    """Aggregate sessions by their local start date for the requested range."""
    if days < 1:
        raise ValueError("days must be at least 1")

    first_day = today - timedelta(days=days - 1)
    totals: dict[date, list[int]] = {}
    for started_at, total_frames, slouch_frames in sessions:
        session_date = datetime.fromisoformat(started_at).astimezone().date()
        if first_day <= session_date <= today:
            values = totals.setdefault(session_date, [0, 0])
            values[0] += total_frames
            values[1] += slouch_frames

    result = []
    for offset in range(days):
        day = first_day + timedelta(days=offset)
        total_frames, slouch_frames = totals.get(day, [0, 0])
        result.append(
            {
                "date": day,
                "slouch_percent": (
                    slouch_frames / total_frames * 100.0 if total_frames else None
                ),
                "total_frames": total_frames,
                "slouch_frames": slouch_frames,
            }
        )
    return result


def weekly_summary(
    sessions: list[tuple[str, int, int]],
    completed_breaks: list[str],
    today: date,
) -> dict[str, float | int]:
    """Calculate current/previous week weighted posture rates and streak."""
    week_start = today - timedelta(days=today.weekday())
    previous_week_start = week_start - timedelta(days=7)
    current_frames = current_slouch_frames = 0
    previous_frames = previous_slouch_frames = 0
    daily = daily_session_stats(sessions, 8 + today.weekday(), today)

    for started_at, total_frames, slouch_frames in sessions:
        session_date = datetime.fromisoformat(started_at).astimezone().date()
        if week_start <= session_date <= today:
            current_frames += total_frames
            current_slouch_frames += slouch_frames
        elif previous_week_start <= session_date < week_start:
            previous_frames += total_frames
            previous_slouch_frames += slouch_frames

    current_percent = (
        current_slouch_frames / current_frames * 100.0 if current_frames else 0.0
    )
    previous_percent = (
        previous_slouch_frames / previous_frames * 100.0
        if previous_frames
        else 0.0
    )

    streak = 0
    by_date = {item["date"]: item["slouch_percent"] for item in daily}
    streak_day = today if by_date[today] is not None else today - timedelta(days=1)
    while streak_day in by_date:
        percent = by_date[streak_day]
        if percent is None or percent >= 25.0:
            break
        streak += 1
        streak_day -= timedelta(days=1)

    completed_this_week = sum(
        1
        for timestamp in completed_breaks
        if week_start
        <= datetime.fromisoformat(timestamp).astimezone().date()
        <= today
    )
    return {
        "slouch_percent_this_week": current_percent,
        "slouch_percent_previous_week": previous_percent,
        "slouch_change_percent_points": current_percent - previous_percent,
        "completed_breaks": completed_this_week,
        "streak_days": streak,
    }
