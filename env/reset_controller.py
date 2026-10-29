"""Bounded retries for an explicitly selected spell-practice task."""
import time


class ResetError(RuntimeError):
    pass


class ResetController:
    def __init__(self, reader, controls, spell_retry=False, timeout=10.0,
                 clock=time.monotonic, sleep=time.sleep):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.reader, self.controls = reader, controls
        self.spell_retry, self.timeout = spell_retry, timeout
        self.clock, self.sleep = clock, sleep

    def reset(self, first=False):
        self.controls.release_all()
        deadline = self.clock() + self.timeout
        confirmed = False
        initial_frame = None
        while self.clock() < deadline:
            self.controls.check_focus()
            sample = self.reader.snapshot()
            if first and sample.ready:
                if initial_frame is None:
                    initial_frame = sample.frame
                elif sample.frame != initial_frame:
                    return sample
            if sample.retry and not confirmed:
                if not self.spell_retry:
                    raise ResetError("Enter spell practice manually or explicitly enable spell retry")
                self.controls.tap_confirm(self.sleep)
                confirmed = True
                initial_frame = sample.frame
            elif confirmed and sample.ready and sample.frame != initial_frame:
                return sample
            elif not first and not confirmed and sample.ready:
                raise ResetError("Task is still running; a new episode was not started")
            elif sample.scene in (1, 6, 7, 10):
                raise ResetError("Manual task selection is required from this screen")
            self.sleep(0.01)
        raise ResetError("Timed out waiting for a playable spell-practice task")
