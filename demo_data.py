"""Seed realistic local monitoring records for a populated dashboard demo."""

import random
from datetime import date, datetime, time, timedelta

from storage import SessionStorage


def load_demo_data(
    storage: SessionStorage, today: date | None = None
) -> int:
    """Add one sample session for each missing day in the last two weeks."""
    current_time = datetime.now().astimezone()
    current_day = today or current_time.date()
    randomizer = random.Random(20261016)
    added = 0
    for days_ago in range(13, -1, -1):
        session_day = current_day - timedelta(days=days_ago)
        if storage.has_session_on(session_day):
            continue
        duration_sec = randomizer.randint(25, 65) * 60
        if session_day == current_day and current_day == current_time.date():
            day_start = datetime.combine(
                session_day, time.min, tzinfo=current_time.tzinfo
            )
            elapsed_today = int((current_time - day_start).total_seconds())
            duration_sec = min(duration_sec, elapsed_today)
            started_at = current_time - timedelta(seconds=duration_sec)
        else:
            started_at = datetime.combine(
                session_day,
                time(
                    hour=randomizer.choice((9, 10, 13, 14)),
                    minute=randomizer.randrange(60),
                ),
            ).astimezone()
        total_frames = duration_sec
        slouch_percent = randomizer.uniform(8.0, 34.0)
        slouch_frames = round(total_frames * slouch_percent / 100.0)
        session_id = storage.add_session(
            started_at, duration_sec, total_frames, slouch_frames
        )
        break_count = (
            randomizer.randint(0, 2) if duration_sec >= 11 * 60 else 0
        )
        for _ in range(break_count):
            completed_at = started_at + timedelta(
                minutes=randomizer.randint(10, duration_sec // 60 - 1)
            )
            storage.log_event("break_started", completed_at, session_id)
            storage.log_event(
                "break_completed",
                completed_at + timedelta(seconds=60),
                session_id,
            )
        added += 1
    return added
