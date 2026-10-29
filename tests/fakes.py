"""Small deterministic substitutes for live dependencies."""
from dataclasses import replace
from env.memory_reader import Snapshot


def sample(**changes):
    return replace(Snapshot(10, 2, 2, 2, 3, 0, False, False, 0, 192.0, 400.0), **changes)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeReader:
    def __init__(self, *samples):
        self.samples = iter(samples)
        self.last = samples[-1]
        self.closed = False

    def snapshot(self):
        value = next(self.samples, self.last)
        if isinstance(value, BaseException):
            raise value
        self.last = value
        return value

    def close(self):
        self.closed = True


class FakeKeyboard:
    def __init__(self):
        self.focus = True
        self.events = []

    def is_foreground(self):
        return self.focus

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))
