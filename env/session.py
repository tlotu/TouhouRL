"""Framework-independent single-life task loop."""
import time

from .frame_stepper import FrameStepper, FrameTimeout
from .reset_controller import ResetController

ENEMY_CAPACITY = 16
OBSERVATION_SCHEMA = "touhourl-state-v2"


def observation(sample, capacity=64):
    def clip(value):
        return max(-1.0, min(1.0, value))

    nearest = sorted(sample.bullets, key=lambda b: (b.x - sample.x)**2 + (b.y - sample.y)**2)
    rows = [[clip((b.x - sample.x) / 384), clip((b.y - sample.y) / 448),
             clip(b.vx / 16), clip(b.vy / 16), b.radius / 32,
             b.state / 4, 1.0] for b in nearest[:capacity]]
    rows.extend([[0.0] * 7 for _ in range(capacity - len(rows))])
    enemies = sorted(sample.enemies, key=lambda e: (
        not bool(e.flags2 & 8), bool(e.flags3 & 8), not bool(e.flags2 & 2),
        (e.x - sample.x)**2 + (e.y - sample.y)**2))
    targets = [[clip((e.x - sample.x) / 384), clip((e.y - sample.y) / 448),
                float(bool(e.flags2 & 8)), float(bool(e.flags2 & 2)),
                float(not bool(e.flags3 & 8)), 1.0] for e in enemies[:ENEMY_CAPACITY]]
    targets.extend([[0.0] * 6 for _ in range(ENEMY_CAPACITY - len(targets))])
    return {"player": [clip(sample.x / 192 - 1), clip(sample.y / 224 - 1),
                       min(sample.lives / 8, 1), min(sample.bombs / 8, 1)],
            "bullets": rows, "enemies": targets}


class GameSession:
    """Real-time diagnostic environment; network-update pausing is not implemented."""

    def __init__(self, reader, controls, repeat=2, capacity=64, max_frames=3600,
                 spell_retry=False, clock=time.monotonic, sleep=time.sleep, quiet_limit=0):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.reader, self.controls = reader, controls
        if not isinstance(quiet_limit, int) or quiet_limit < 0:
            raise ValueError("quiet_limit must be a nonnegative integer")
        self.quiet_limit = quiet_limit
        self.quiet_frames = 0
        self.saw_bullets = False
        self.capacity, self.max_frames = capacity, max_frames
        self.clock = clock
        self.sleep = sleep
        self.stepper = FrameStepper(reader, controls, repeat, clock=clock, sleep=sleep)
        self.resetter = ResetController(reader, controls, spell_retry, clock=clock, sleep=sleep)
        self.sample = None
        self.active = False
        self.started = False
        self.closed = False
        self.elapsed_frames = 0

    def _info(self, reason=None, frames=0, seconds=0.0):
        s = self.sample
        return {"reason": reason, "frame": s.frame, "scene": s.scene,
                "lives": s.lives, "bombs": s.bombs, "deaths": s.deaths,
                "dying": s.dying, "retry": s.retry,
                "bullet_count": len(s.bullets),
                "enemy_count": len(s.enemies),
                "dropped_enemies": max(0, len(s.enemies) - ENEMY_CAPACITY),
                "enemies_sampled": s.running,
                "enemies": [enemy.record() for enemy in s.enemies],
                "result_menu": ({"active": s.result_menu.active, "selection": s.result_menu.selection,
                                 "phase": s.result_menu.phase, "outcome_flag": s.result_menu.outcome_flag}
                                if s.result_menu is not None else None),
                "radius_fallback_count": sum(b.radius_estimated for b in s.bullets),
                "radius_fallback_slots": [b.slot for b in s.bullets if b.radius_estimated],
                "dropped_bullets": max(0, len(s.bullets) - self.capacity),
                "quiet_frames": self.quiet_frames,
                "delta_frames": frames, "elapsed_frames": self.elapsed_frames,
                "step_seconds": seconds}

    def reset(self):
        if self.closed:
            raise RuntimeError("Session is closed")
        was_active = self.active
        self.active = False
        try:
            if was_active:
                raise RuntimeError("Finish the current episode before resetting")
            self.sample = self.resetter.reset(first=not self.started)
            self.started = self.active = True
            self.elapsed_frames = 0
            self.quiet_frames = 0
            self.saw_bullets = bool(self.sample.bullets)
            self.last_observation_time = self.clock()
            return observation(self.sample, self.capacity), self._info()
        except BaseException:
            self.controls.release_all()
            raise

    def step(self, action):
        if not self.active or self.closed:
            raise RuntimeError("Call reset() before stepping")
        try:
            # The game keeps running between calls; long gaps invalidate the sample.
            if self.clock() - self.last_observation_time > 0.25:
                raise FrameTimeout("Too much time passed since the last observation")
            self.controls.apply(action)
            previous = self.sample
            self.sample, frames, seconds = self.stepper.advance(previous)
            self.elapsed_frames += frames
            if self.sample.result_menu is not None and (
                    self.sample.result_menu.active or self.sample.retry
                    or self.sample.deaths > previous.deaths):
                self.controls.release_all()
                deadline = self.clock() + 3.0
                saw_result = self.sample.result_menu.active
                while self.sample.result_menu.outcome is None:
                    self.controls.check_focus()
                    if self.clock() >= deadline:
                        raise RuntimeError("Result menu did not settle into an interpretable phase")
                    self.sleep(0.01)
                    self.sample = self.reader.snapshot()
                    if self.sample.result_menu is None or (saw_result and not self.sample.result_menu.active):
                        raise RuntimeError("Result menu disappeared before outcome verification")
                    saw_result = saw_result or self.sample.result_menu.active
            if self.sample.bullets:
                self.saw_bullets = True
                self.quiet_frames = 0
            elif self.saw_bullets:
                self.quiet_frames += frames
            died = self.sample.deaths > previous.deaths or self.sample.retry
            outcome = self.sample.result_menu.outcome if self.sample.result_menu else None
            terminated = outcome is not None or died or not self.sample.running
            quiet_stop = bool(self.quiet_limit and self.quiet_frames >= self.quiet_limit)
            truncated = not terminated and (quiet_stop or self.elapsed_frames >= self.max_frames)
            reason = ("spell_success" if outcome == "success" else
                      "spell_failure" if outcome == "failure" else
                      "death" if died else "scene_exit" if terminated else
                      "quiet_after_combat" if quiet_stop else "time_limit" if truncated else None)
            # Terminal frames do not earn survival credit.
            reward = (-1.0 if died or outcome == "failure" else 0.0) if terminated else frames / 60.0
            if terminated or truncated:
                self.active = False
                self.controls.release_all()
            self.last_observation_time = self.clock()
            return observation(self.sample, self.capacity), reward, terminated, truncated, self._info(reason, frames, seconds)
        except BaseException:
            self.active = False
            self.controls.release_all()
            raise

    def close(self):
        if self.closed:
            return
        self.active = False
        try:
            self.controls.release_all()
        finally:
            self.reader.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
