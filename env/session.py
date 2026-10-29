"""Framework-independent single-life task loop."""
import time

from .frame_stepper import FrameStepper, FrameTimeout
from .reset_controller import ResetController


def observation(sample, capacity=64):
    def clip(value):
        return max(-1.0, min(1.0, value))

    nearest = sorted(sample.bullets, key=lambda b: (b.x - sample.x)**2 + (b.y - sample.y)**2)
    rows = [[clip((b.x - sample.x) / 384), clip((b.y - sample.y) / 448),
             clip(b.vx / 16), clip(b.vy / 16), b.radius / 32,
             b.state / 4, 1.0] for b in nearest[:capacity]]
    rows.extend([[0.0] * 7 for _ in range(capacity - len(rows))])
    return {"player": [clip(sample.x / 192 - 1), clip(sample.y / 224 - 1),
                       min(sample.lives / 8, 1), min(sample.bombs / 8, 1)],
            "bullets": rows}


class GameSession:
    """Real-time diagnostic environment; network-update pausing is not implemented."""

    def __init__(self, reader, controls, repeat=2, capacity=64, max_frames=3600,
                 spell_retry=False, clock=time.monotonic, sleep=time.sleep):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self.reader, self.controls = reader, controls
        self.capacity, self.max_frames = capacity, max_frames
        self.clock = clock
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
                "dropped_bullets": max(0, len(s.bullets) - self.capacity),
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
            died = self.sample.deaths > previous.deaths or self.sample.retry
            terminated = died or not self.sample.running
            truncated = not terminated and self.elapsed_frames >= self.max_frames
            reason = "death" if died else "scene_exit" if terminated else "time_limit" if truncated else None
            # Terminal frames do not earn survival credit.
            reward = (-1.0 if died else 0.0) if terminated else frames / 60.0
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
