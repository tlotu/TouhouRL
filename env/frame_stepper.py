"""Measure real-time frames; this is not an engine step hook."""
import time


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
        while self.clock() - started < self.timeout:
            self.controls.check_focus()
            read_started = self.clock()
            sample = self.reader.snapshot()
            read_finished = self.clock()
            reads += 1
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
                                    "sample_read_seconds": read_finished - read_started,
                                    "sample_received_at": read_finished}
                return sample, elapsed, self.clock() - started
            self.sleep(0.001)
        raise FrameTimeout("Game frames stalled; check pause state and focus")
