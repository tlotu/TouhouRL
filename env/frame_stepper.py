"""Measure real-time frames; this is not an engine step hook."""
import time
from .memory_reader import SnapshotRaceError


ACQUISITION_CONTRACT = "same-frame-bounded-retry-v1"


class FrameTimeout(RuntimeError):
    pass


class FrameStepper:
    def __init__(self, reader, controls, repeat=2, timeout=0.5,
                 clock=time.perf_counter, sleep=time.sleep):
        if not isinstance(repeat, int) or not 1 <= repeat <= 16:
            raise ValueError("repeat must be an integer from 1 to 16")
        if not 0 < timeout <= 0.5:
            raise ValueError("timeout must be in (0, 0.5]")
        self.reader, self.controls = reader, controls
        self.repeat, self.timeout = repeat, timeout
        self.clock, self.sleep = clock, sleep
        self.last_timing = {}

    def advance(self, previous):
        started = self.clock()
        last = previous.frame
        elapsed = 0
        reads = 0
        read_seconds = 0.0
        race_retries = 0
        last_attempt_raced = False
        while self.clock() - started < self.timeout:
            self.controls.check_focus()
            read_started = self.clock()
            reads += 1
            try:
                sample = self.reader.snapshot()
            except SnapshotRaceError:
                last_attempt_raced = True
                race_retries += 1
                read_seconds += self.clock() - read_started
                if self.clock() - started >= self.timeout:
                    raise FrameTimeout("State acquisition exceeded the step deadline during frame-race retries")
                self.sleep(0.001)
                continue
            read_finished = self.clock()
            last_attempt_raced = False
            read_seconds += read_finished - read_started
            if self.clock() - started >= self.timeout:
                raise FrameTimeout("State acquisition exceeded the step deadline")
            delta = (sample.frame - last) & 255
            if delta > 32:
                raise FrameTimeout("Frame counter jumped or reset")
            elapsed += delta
            last = sample.frame
            if not sample.running or sample.deaths > previous.deaths or elapsed >= self.repeat:
                self.last_timing = {"snapshot_calls": reads, "snapshot_seconds": read_seconds,
                                    "snapshot_race_retries": race_retries,
                                    "sample_read_seconds": read_finished - read_started,
                                    "sample_received_at": read_finished}
                return sample, elapsed, self.clock() - started
            self.sleep(0.001)
        if last_attempt_raced:
            raise FrameTimeout("State acquisition exceeded the step deadline during frame-race retries")
        raise FrameTimeout("Game frames stalled; check pause state and focus")
