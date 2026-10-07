"""Debounced posture episode and alert timing state machine."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from config import (
    MAX_UNOBSERVED_SECONDS,
    GRACE_PERIOD_SECONDS,
    MIN_BAD_POSE_SECONDS,
    NOTIFICATION_SECONDS,
    POSTURE_DEVIATION_THRESHOLD_DEGREES,
    SOFT_ALERT_SECONDS,
)


@dataclass(frozen=True)
class AlertUpdate:
    """State transitions produced by one posture observation."""

    episode_started: datetime | None = None
    episode_ended: datetime | None = None
    soft_alert: bool = False
    notification: bool = False


class AlertManager:
    """Debounce posture transitions without losing a confirmed episode timer.

    The state machine moves between normal and poor posture only after the
    candidate state lasts for the grace period. A normal-pose candidate during
    a poor-posture episode keeps that episode and its original start time alive;
    returning to poor posture cancels the candidate, while five continuous
    seconds of normal posture end the episode at the start of that recovery.
    This also rejects brief poor-posture spikes while the confirmed state is
    normal.
    """

    def __init__(
        self,
        deviation_threshold: float = POSTURE_DEVIATION_THRESHOLD_DEGREES,
        minimum_bad_seconds: float = MIN_BAD_POSE_SECONDS,
        soft_alert_seconds: float = SOFT_ALERT_SECONDS,
        notification_seconds: float = NOTIFICATION_SECONDS,
        max_unobserved_seconds: float = MAX_UNOBSERVED_SECONDS,
        grace_period_seconds: float = GRACE_PERIOD_SECONDS,
    ) -> None:
        """Create a posture alert state machine.

        Args:
            deviation_threshold: Maximum allowed absolute angle deviation.
            minimum_bad_seconds: Debounce duration before opening an episode.
            soft_alert_seconds: Delay before changing the tray icon state.
            notification_seconds: Delay before sending a system notification.
            max_unobserved_seconds: Gap after which a lost pose ends tracking.
            grace_period_seconds: Continuous candidate duration required before
                changing the confirmed posture state in either direction.

        Returns:
            None.
        """
        self.deviation_threshold = deviation_threshold
        self.minimum_bad_seconds = minimum_bad_seconds
        self.soft_alert_seconds = soft_alert_seconds
        self.notification_seconds = notification_seconds
        self.max_unobserved_seconds = max_unobserved_seconds
        self.grace_period_seconds = grace_period_seconds
        self._bad_since: datetime | None = None
        self._last_observed: datetime | None = None
        self._episode_active = False
        self._grace_since: datetime | None = None
        self._grace_target_bad: bool | None = None
        self._timers_paused_since: datetime | None = None
        self._soft_sent = False
        self._notification_sent = False

    def update(
        self,
        deviation_degrees: float | None,
        observed_at: datetime | None = None,
        posture_bad: bool | None = None,
    ) -> AlertUpdate:
        """Consume a posture observation and return any state transitions.

        Args:
            deviation_degrees: Largest absolute baseline deviation, or None if no pose.
            observed_at: Observation time; local aware time is used by default.
            posture_bad: Explicit poor-posture decision from multiple metric
                thresholds. When omitted, the numeric deviation threshold is used.

        Returns:
            Episode start/end and one-shot alert events for this observation.
        """
        current_time = observed_at or datetime.now().astimezone()
        if deviation_degrees is None:
            if (
                self._last_observed is not None
                and (current_time - self._last_observed).total_seconds()
                > self.max_unobserved_seconds
            ):
                return self.reset(ended_at=self._last_observed)
            return AlertUpdate()

        self._resume_timers(current_time)
        self._last_observed = current_time
        deviation_is_bad = (
            deviation_degrees > self.deviation_threshold
            if posture_bad is None
            else posture_bad
        )
        if deviation_is_bad == self._episode_active:
            self._grace_since = None
            self._grace_target_bad = None
            if deviation_is_bad:
                return self._make_bad_update(current_time)
            self._bad_since = None
            self._soft_sent = False
            self._notification_sent = False
            return AlertUpdate()

        if self._grace_since is None or self._grace_target_bad != deviation_is_bad:
            self._grace_since = current_time
            self._grace_target_bad = deviation_is_bad

        grace_elapsed = (current_time - self._grace_since).total_seconds()
        transition_delay = max(self.grace_period_seconds, self.minimum_bad_seconds)
        if grace_elapsed < transition_delay:
            if deviation_is_bad and self._bad_since is None:
                self._bad_since = current_time
            return AlertUpdate()

        if deviation_is_bad:
            if self._bad_since is None:
                self._bad_since = self._grace_since
            self._episode_active = True
            self._grace_since = None
            self._grace_target_bad = None
            return self._make_bad_update(current_time, episode_started=self._bad_since)

        ended_at = self._grace_since
        update = self.reset(ended_at=ended_at)
        return update

    def _make_bad_update(
        self,
        current_time: datetime,
        episode_started: datetime | None = None,
    ) -> AlertUpdate:
        """Build sustained-poor-posture events using the preserved episode start.

        Args:
            current_time: Timestamp of the current bad-pose observation.
            episode_started: Optional newly confirmed episode start time.

        Returns:
            Episode transition and one-shot soft/system alert events.
        """
        if self._bad_since is None:
            self._bad_since = current_time
        elapsed = (current_time - self._bad_since).total_seconds()
        new_episode_started = episode_started
        if not self._episode_active and elapsed >= self.minimum_bad_seconds:
            self._episode_active = True
            new_episode_started = new_episode_started or self._bad_since

        soft_alert = elapsed >= self.soft_alert_seconds and not self._soft_sent
        notification = elapsed >= self.notification_seconds and not self._notification_sent
        if soft_alert:
            self._soft_sent = True
        if notification:
            self._notification_sent = True
        return AlertUpdate(
            episode_started=new_episode_started,
            soft_alert=soft_alert,
            notification=notification,
        )

    def reset(self, ended_at: datetime | None = None) -> AlertUpdate:
        """Clear the current candidate/episode and report its end if active.

        Args:
            ended_at: Timestamp to record for an active episode's end.

        Returns:
            An episode-end event when an episode was active, otherwise an empty update.
        """
        episode_ended = ended_at if self._episode_active else None
        self._bad_since = None
        self._last_observed = None
        self._episode_active = False
        self._grace_since = None
        self._grace_target_bad = None
        self._soft_sent = False
        self._notification_sent = False
        self._timers_paused_since = None
        return AlertUpdate(episode_ended=episode_ended)

    def pause_timers(self, paused_at: datetime | None = None) -> None:
        """Freeze episode and grace clocks while a camera sample is unusable.

        Args:
            paused_at: Timestamp of the first consecutive skipped observation.

        Returns:
            None.
        """
        if self._timers_paused_since is None:
            self._timers_paused_since = paused_at or datetime.now().astimezone()

    def _resume_timers(self, resumed_at: datetime) -> None:
        """Exclude unusable-frame duration from episode and transition clocks.

        Args:
            resumed_at: Timestamp of the next valid posture sample.

        Returns:
            None.
        """
        if self._timers_paused_since is None:
            return
        paused_seconds = max(
            (resumed_at - self._timers_paused_since).total_seconds(), 0.0
        )
        pause_delta = timedelta(seconds=paused_seconds)
        if self._bad_since is not None:
            self._bad_since += pause_delta
        if self._grace_since is not None:
            self._grace_since += pause_delta
        if self._last_observed is not None:
            self._last_observed += pause_delta
        self._timers_paused_since = None

    def deviation_duration(self, observed_at: datetime | None = None) -> float:
        """Return how long the current above-threshold deviation has lasted.

        Args:
            observed_at: Timestamp at which to calculate elapsed time.

        Returns:
            Elapsed seconds, or zero when no deviation is currently tracked.
        """
        if self._bad_since is None:
            return 0.0
        current_time = self._timers_paused_since or observed_at or datetime.now().astimezone()
        return max((current_time - self._bad_since).total_seconds(), 0.0)

    @property
    def posture_bad(self) -> bool:
        """Return the confirmed posture state, excluding pending transitions."""
        return self._episode_active

    @property
    def grace_active(self) -> bool:
        """Return whether either direction of posture transition is pending."""
        return self._grace_since is not None

    def grace_remaining(self, observed_at: datetime | None = None) -> float:
        """Return the remaining candidate time before the posture state changes.

        Args:
            observed_at: Timestamp to use; current local time is the default.

        Returns:
            Remaining grace seconds, or zero when no transition is pending.
        """
        if self._grace_since is None:
            return 0.0
        current_time = self._timers_paused_since or observed_at or datetime.now().astimezone()
        transition_delay = max(self.grace_period_seconds, self.minimum_bad_seconds)
        elapsed = (current_time - self._grace_since).total_seconds()
        return max(transition_delay - elapsed, 0.0)
