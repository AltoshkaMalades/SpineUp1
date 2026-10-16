"""Local SQLite storage for monitoring sessions, sampled frames, and events."""

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Literal

from config import SESSION_DATABASE_PATH
from stats import daily_session_stats, weekly_summary


EventType = Literal[
    "slouch_alert", "posture_corrected", "break_started", "break_completed"
]
EVENT_TYPES = frozenset(
    {"slouch_alert", "posture_corrected", "break_started", "break_completed"}
)
FRAME_FLUSH_BATCH_SIZE = 30


class SessionStorage:
    """Persist monitoring data while batching sampled frame updates in memory."""

    def __init__(self, database_path: Path = SESSION_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.database_path, check_same_thread=False
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_sec REAL,
                total_frames INTEGER NOT NULL DEFAULT 0,
                slouch_frames INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_started_at
                ON sessions(started_at);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(id),
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL CHECK (
                    type IN (
                        'slouch_alert',
                        'posture_corrected',
                        'break_started',
                        'break_completed'
                    )
                )
            );
            CREATE INDEX IF NOT EXISTS idx_events_timestamp_type
                ON events(timestamp, type);
            """
        )
        self._connection.commit()
        self._lock = RLock()
        self._active_session_id: int | None = None
        self._pending_total_frames = 0
        self._pending_slouch_frames = 0

        stale_ended_at = datetime.now().astimezone()
        with self._connection:
            self._connection.execute(
                """UPDATE sessions
                   SET ended_at = ?, duration_sec =
                       MAX(0, (julianday(?) - julianday(started_at)) * 86400)
                   WHERE ended_at IS NULL""",
                (stale_ended_at.isoformat(), stale_ended_at.isoformat()),
            )

    def start_session(self) -> int:
        """Create and activate a new monitoring session."""
        with self._lock:
            if self._active_session_id is not None:
                self.end_session()
            started_at = datetime.now().astimezone()
            with self._connection:
                cursor = self._connection.execute(
                    "INSERT INTO sessions (started_at) VALUES (?)",
                    (started_at.isoformat(),),
                )
            self._active_session_id = int(cursor.lastrowid)
            self._pending_total_frames = 0
            self._pending_slouch_frames = 0
            return self._active_session_id

    def end_session(self) -> int | None:
        """Flush the active session's frame batch and close its duration."""
        with self._lock:
            session_id = self._active_session_id
            if session_id is None:
                return None
            ended_at = datetime.now().astimezone()
            with self._connection:
                self._connection.execute(
                    """UPDATE sessions
                       SET ended_at = ?, duration_sec =
                           MAX(0, (julianday(?) - julianday(started_at)) * 86400),
                           total_frames = total_frames + ?,
                           slouch_frames = slouch_frames + ?
                       WHERE id = ?""",
                    (
                        ended_at.isoformat(),
                        ended_at.isoformat(),
                        self._pending_total_frames,
                        self._pending_slouch_frames,
                        session_id,
                    ),
                )
            self._active_session_id = None
            self._pending_total_frames = 0
            self._pending_slouch_frames = 0
            return session_id

    def log_event(
        self,
        event_type: EventType,
        timestamp: datetime | None = None,
        session_id: int | None = None,
    ) -> None:
        """Record a validated event, linked to the current session by default."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type}")
        event_time = (timestamp or datetime.now().astimezone()).astimezone()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO events (session_id, timestamp, type) VALUES (?, ?, ?)",
                (
                    self._active_session_id if session_id is None else session_id,
                    event_time.isoformat(),
                    event_type,
                ),
            )

    def log_frame(self, is_slouching: bool) -> None:
        """Count a valid posture sample and flush in batches, not per frame."""
        with self._lock:
            if self._active_session_id is None:
                return
            self._pending_total_frames += 1
            if is_slouching:
                self._pending_slouch_frames += 1
            if self._pending_total_frames >= FRAME_FLUSH_BATCH_SIZE:
                self._flush_frames()

    def _flush_frames(self) -> None:
        if self._active_session_id is None or not self._pending_total_frames:
            return
        with self._connection:
            self._connection.execute(
                """UPDATE sessions
                   SET total_frames = total_frames + ?,
                       slouch_frames = slouch_frames + ?
                   WHERE id = ?""",
                (
                    self._pending_total_frames,
                    self._pending_slouch_frames,
                    self._active_session_id,
                ),
            )
        self._pending_total_frames = 0
        self._pending_slouch_frames = 0

    def get_daily_stats(self, days: int = 7) -> list[dict[str, object]]:
        """Return weighted daily slouch percentages and frame totals."""
        if days < 1:
            raise ValueError("days must be at least 1")
        cutoff = (date.today() - timedelta(days=days - 1)).isoformat()
        with self._lock:
            if self._active_session_id is None:
                rows = self._connection.execute(
                    """SELECT started_at, total_frames, slouch_frames
                       FROM sessions WHERE started_at >= ?""",
                    (cutoff,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """SELECT started_at, total_frames, slouch_frames
                       FROM sessions WHERE started_at >= ? AND id != ?""",
                    (cutoff, self._active_session_id),
                ).fetchall()
            active = self._pending_session_rows()
        return daily_session_stats(rows + active, days, date.today())

    def get_weekly_summary(self) -> dict[str, float | int]:
        """Return this/last week's slouch rates, breaks, and low-slouch streak."""
        lower_bound = (
            date.today() - timedelta(days=date.today().weekday() + 7)
        ).isoformat()
        with self._lock:
            if self._active_session_id is None:
                sessions = self._connection.execute(
                    """SELECT started_at, total_frames, slouch_frames
                       FROM sessions WHERE started_at >= ?""",
                    (lower_bound,),
                ).fetchall()
            else:
                sessions = self._connection.execute(
                    """SELECT started_at, total_frames, slouch_frames
                       FROM sessions WHERE started_at >= ? AND id != ?""",
                    (lower_bound, self._active_session_id),
                ).fetchall()
            sessions.extend(self._pending_session_rows())
            break_rows = self._connection.execute(
                "SELECT timestamp FROM events WHERE type = 'break_completed'"
            ).fetchall()
        return weekly_summary(
            sessions, [row[0] for row in break_rows], date.today()
        )

    def get_recent_sessions(self, limit: int = 10) -> list[dict[str, object]]:
        """Return the most recently started sessions with computed slouch rates."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        with self._lock:
            rows = self._connection.execute(
                """SELECT id, started_at, ended_at, duration_sec,
                          total_frames, slouch_frames
                   FROM sessions ORDER BY started_at DESC, id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            active = self._pending_session_rows()
            active_id = self._active_session_id
        results = [
            {
                "id": row[0],
                "started_at": row[1],
                "ended_at": row[2],
                "duration_sec": row[3],
                "total_frames": row[4],
                "slouch_frames": row[5],
                "slouch_percent": (
                    row[5] / row[4] * 100.0 if row[4] else None
                ),
            }
            for row in rows
        ]
        if active:
            total, slouch = active[0][1], active[0][2]
            active_result = {
                "id": active_id,
                "started_at": active[0][0],
                "ended_at": None,
                "duration_sec": None,
                "total_frames": total,
                "slouch_frames": slouch,
                "slouch_percent": slouch / total * 100.0 if total else None,
            }
            existing_index = next(
                (
                    index
                    for index, result in enumerate(results)
                    if result["id"] == active_id
                ),
                None,
            )
            if existing_index is None:
                results.append(active_result)
            else:
                results[existing_index] = active_result
            results.sort(key=lambda item: item["started_at"], reverse=True)
            results = results[:limit]
        return results

    def add_session(
        self,
        started_at: datetime,
        duration_sec: int,
        total_frames: int,
        slouch_frames: int,
    ) -> int:
        """Insert a completed historical session for the demo-data helper."""
        if duration_sec < 0 or total_frames < 0 or not 0 <= slouch_frames <= total_frames:
            raise ValueError("Invalid demo session values")
        ended_at = started_at + timedelta(seconds=duration_sec)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO sessions
                   (started_at, ended_at, duration_sec, total_frames, slouch_frames)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    started_at.astimezone().isoformat(),
                    ended_at.astimezone().isoformat(),
                    duration_sec,
                    total_frames,
                    slouch_frames,
                ),
            )
            return int(cursor.lastrowid)

    def has_session_on(self, session_date: date) -> bool:
        """Check whether any completed or active session starts on a date."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT started_at FROM sessions"
            ).fetchall()
        return any(
            datetime.fromisoformat(row[0]).astimezone().date() == session_date
            for row in rows
        )

    def close(self) -> None:
        """End any open session and close the SQLite connection."""
        with self._lock:
            try:
                self.end_session()
            finally:
                self._connection.close()

    def _pending_session_rows(self) -> list[tuple[str, int, int]]:
        if self._active_session_id is None:
            return []
        row = self._connection.execute(
            """SELECT started_at, total_frames, slouch_frames
               FROM sessions WHERE id = ?""",
            (self._active_session_id,),
        ).fetchone()
        if row is None:
            return []
        return [
            (
                row[0],
                row[1] + self._pending_total_frames,
                row[2] + self._pending_slouch_frames,
            )
        ]


_default_storage: SessionStorage | None = None


def initialize_storage(
    database_path: Path = SESSION_DATABASE_PATH,
) -> SessionStorage:
    """Initialize the module-level storage used by the requested convenience API."""
    global _default_storage
    if _default_storage is not None:
        _default_storage.close()
    _default_storage = SessionStorage(database_path)
    return _default_storage


def _get_default_storage() -> SessionStorage:
    global _default_storage
    if _default_storage is None:
        _default_storage = SessionStorage()
    return _default_storage


def start_session() -> int:
    return _get_default_storage().start_session()


def end_session() -> int | None:
    return _get_default_storage().end_session()


def log_event(
    event_type: EventType,
    timestamp: datetime | None = None,
    session_id: int | None = None,
) -> None:
    _get_default_storage().log_event(event_type, timestamp, session_id)


def log_frame(is_slouching: bool) -> None:
    _get_default_storage().log_frame(is_slouching)


def get_daily_stats(days: int = 7) -> list[dict[str, object]]:
    return _get_default_storage().get_daily_stats(days)


def get_weekly_summary() -> dict[str, float | int]:
    return _get_default_storage().get_weekly_summary()


def get_recent_sessions(limit: int = 10) -> list[dict[str, object]]:
    return _get_default_storage().get_recent_sessions(limit)


def close_storage() -> None:
    global _default_storage
    if _default_storage is not None:
        _default_storage.close()
        _default_storage = None
