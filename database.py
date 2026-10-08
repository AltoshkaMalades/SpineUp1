"""SQLite persistence and daily poor-posture statistics."""

import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from threading import Lock

from config import DATABASE_PATH


class PostureDatabase:
    """Store posture episodes in a local SQLite database."""

    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        """Open the local database and create its schema if needed.

        Args:
            database_path: SQLite database file path.

        Returns:
            None.
        """
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS posture_episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT
            )"""
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_posture_episodes_start ON posture_episodes(start_time)"
        )
        self._connection.commit()
        self._lock = Lock()

    def close_stale_episodes(self, ended_at: datetime | None = None) -> None:
        """Close episodes left open by an earlier application session.

        Args:
            ended_at: Recovery timestamp; current local time is used by default.

        Returns:
            None.
        """
        timestamp = (ended_at or datetime.now().astimezone()).isoformat()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE posture_episodes SET end_time = ? WHERE end_time IS NULL",
                (timestamp,),
            )

    def start_episode(self, started_at: datetime) -> int:
        """Insert an open posture episode.

        Args:
            started_at: First timestamp of the sustained posture deviation.

        Returns:
            The SQLite row id for the new episode.
        """
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO posture_episodes (start_time) VALUES (?)",
                (started_at.isoformat(),),
            )
            return int(cursor.lastrowid)

    def end_episode(self, episode_id: int, ended_at: datetime) -> None:
        """Set the end time for a previously opened episode.

        Args:
            episode_id: SQLite row id returned by start_episode.
            ended_at: Timestamp at which posture returned to baseline.

        Returns:
            None.
        """
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE posture_episodes SET end_time = ? WHERE id = ?",
                (ended_at.isoformat(), episode_id),
            )

    def daily_poor_posture_percent(
        self, on_date: date | None = None, now: datetime | None = None
    ) -> float:
        """Calculate today's share of elapsed time recorded with poor posture.

        Args:
            on_date: Local calendar date to summarize; defaults to today's date.
            now: End of the measured period; defaults to current local time.

        Returns:
            Poor-posture percentage from 0.0 to 100.0.
        """
        current_time = now or datetime.now().astimezone()
        current_date = on_date or current_time.date()
        day_start = datetime.combine(current_date, time.min, tzinfo=current_time.tzinfo)
        period_end = min(current_time, day_start + timedelta(days=1))
        elapsed_seconds = max((period_end - day_start).total_seconds(), 0.0)
        if elapsed_seconds == 0.0:
            return 0.0

        with self._lock:
            episodes = self._connection.execute(
                "SELECT start_time, end_time FROM posture_episodes"
            ).fetchall()

        poor_seconds = 0.0
        for start_text, end_text in episodes:
            episode_start = datetime.fromisoformat(start_text)
            episode_end = (
                datetime.fromisoformat(end_text) if end_text else current_time
            )
            if episode_start.tzinfo is None:
                episode_start = episode_start.astimezone()
            if episode_end.tzinfo is None:
                episode_end = episode_end.astimezone()
            overlap_start = max(episode_start, day_start)
            overlap_end = min(episode_end, period_end)
            poor_seconds += max((overlap_end - overlap_start).total_seconds(), 0.0)
        return min(100.0, poor_seconds / elapsed_seconds * 100.0)

    def daily_episodes(
        self, on_date: date | None = None, now: datetime | None = None
    ) -> list[tuple[datetime, datetime | None]]:
        """Return episode intervals that overlap the requested local calendar day.

        Args:
            on_date: Local date to list; defaults to today's date.
            now: End time used to clip open episodes; defaults to current time.

        Returns:
            Episode start/end pairs clipped to the selected day. An open current-day
            episode has an end value of None.
        """
        current_time = now or datetime.now().astimezone()
        current_date = on_date or current_time.date()
        day_start = datetime.combine(current_date, time.min, tzinfo=current_time.tzinfo)
        period_end = min(current_time, day_start + timedelta(days=1))
        with self._lock:
            rows = self._connection.execute(
                "SELECT start_time, end_time FROM posture_episodes ORDER BY start_time"
            ).fetchall()

        episodes: list[tuple[datetime, datetime | None]] = []
        for start_text, end_text in rows:
            started_at = datetime.fromisoformat(start_text)
            ended_at = datetime.fromisoformat(end_text) if end_text else None
            if started_at.tzinfo is None:
                started_at = started_at.astimezone()
            effective_end = ended_at or current_time
            if effective_end.tzinfo is None:
                effective_end = effective_end.astimezone()
            overlap_start = max(started_at, day_start)
            overlap_end = min(effective_end, period_end)
            if overlap_end > overlap_start:
                episodes.append(
                    (overlap_start, None if ended_at is None else overlap_end)
                )
        return episodes


    def close(self) -> None:
        """Close the SQLite connection.

        Returns:
            None.
        """
        with self._lock:
            self._connection.close()
