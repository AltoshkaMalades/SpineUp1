"""Tests for session/event persistence and statistics aggregation."""

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

from demo_data import load_demo_data
from storage import (
    SessionStorage,
    close_storage,
    get_recent_sessions,
    initialize_storage,
    log_frame,
    start_session,
    end_session,
)


class SessionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.storage = SessionStorage(Path(self.temporary_directory.name) / "test.db")

    def tearDown(self) -> None:
        self.storage.close()
        self.temporary_directory.cleanup()

    def test_frames_are_batched_and_flushed_when_session_ends(self) -> None:
        session_id = self.storage.start_session()
        for index in range(29):
            self.storage.log_frame(index % 2 == 0)
        row = self.storage._connection.execute(
            "SELECT total_frames, slouch_frames FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        self.assertEqual(row, (0, 0))

        self.storage.log_frame(True)
        row = self.storage._connection.execute(
            "SELECT total_frames, slouch_frames FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        self.assertEqual(row, (30, 16))
        self.storage.log_frame(True)
        self.storage.end_session()
        row = self.storage._connection.execute(
            """SELECT ended_at, duration_sec, total_frames, slouch_frames
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
        self.assertIsNotNone(row[0])
        self.assertGreaterEqual(row[1], 0)
        self.assertEqual(row[2:], (31, 17))

    def test_active_session_counts_include_flushed_and_pending_frames_once(self) -> None:
        self.storage.start_session()
        for index in range(35):
            self.storage.log_frame(index < 7)

        daily = self.storage.get_daily_stats(1)[0]
        summary = self.storage.get_weekly_summary()

        self.assertEqual(daily["total_frames"], 35)
        self.assertEqual(daily["slouch_frames"], 7)
        self.assertEqual(daily["slouch_percent"], 20.0)
        self.assertEqual(summary["slouch_percent_this_week"], 20.0)

    def test_event_types_are_checked_and_linked_to_active_session(self) -> None:
        session_id = self.storage.start_session()
        self.storage.log_event("slouch_alert")
        with self.assertRaises(ValueError):
            self.storage.log_event("unknown")  # type: ignore[arg-type]
        row = self.storage._connection.execute(
            "SELECT session_id, type FROM events"
        ).fetchone()
        self.assertEqual(row, (session_id, "slouch_alert"))

    def test_demo_data_seeds_fourteen_days_only_once(self) -> None:
        today = date.today()
        self.assertEqual(load_demo_data(self.storage, today), 14)
        self.assertEqual(load_demo_data(self.storage, today), 0)
        self.assertEqual(len(self.storage.get_daily_stats(14)), 14)
        self.assertEqual(len(self.storage.get_recent_sessions(20)), 14)

    def test_weekly_summary_uses_frame_weighting_and_completed_breaks(self) -> None:
        today = datetime.now().astimezone()
        week_start = today.date() - timedelta(days=today.weekday())
        current_session = datetime.combine(
            week_start, datetime.min.time(), tzinfo=today.tzinfo
        )
        previous_session = current_session - timedelta(days=7)
        self.storage.add_session(current_session, 600, 100, 20)
        previous_id = self.storage.add_session(previous_session, 600, 100, 40)
        current_break_time = current_session + timedelta(minutes=5)
        self.storage.log_event("break_completed", current_break_time)
        break_time = previous_session + timedelta(minutes=5)
        self.storage.log_event("break_completed", break_time, previous_id)

        summary = self.storage.get_weekly_summary()
        self.assertEqual(summary["slouch_percent_this_week"], 20.0)
        self.assertEqual(summary["slouch_percent_previous_week"], 40.0)
        self.assertEqual(summary["slouch_change_percent_points"], -20.0)
        self.assertEqual(summary["completed_breaks"], 1)

    def test_streak_can_continue_across_the_week_boundary(self) -> None:
        today = date.today()
        weekday = datetime.now().astimezone().weekday()
        expected_streak = 8 + weekday
        for offset in range(expected_streak):
            session_day = today - timedelta(days=offset)
            started_at = datetime.combine(
                session_day, datetime.min.time()
            ).astimezone()
            self.storage.add_session(started_at, 600, 100, 10)

        self.assertEqual(
            self.storage.get_weekly_summary()["streak_days"], expected_streak
        )

    def test_module_level_storage_functions_share_the_initialized_store(self) -> None:
        store_path = Path(self.temporary_directory.name) / "module-api.db"
        initialize_storage(store_path)
        try:
            session_id = start_session()
            log_frame(True)
            self.assertEqual(end_session(), session_id)
            sessions = get_recent_sessions()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["slouch_frames"], 1)
        finally:
            close_storage()


if __name__ == "__main__":
    unittest.main()
