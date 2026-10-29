"""Nine-direction input with explicit key ownership."""
import operator
import time


DIRECTIONS = ((), ("up",), ("up", "right"), ("right",),
              ("down", "right"), ("down",), ("down", "left"),
              ("left",), ("up", "left"))


class InputError(RuntimeError):
    pass


class ActionEmulator:
    """keyboard provides is_foreground(), key_down(), and key_up()."""

    def __init__(self, keyboard, shoot=True, slow=True):
        self.keyboard = keyboard
        self.shoot = shoot
        self.slow = slow
        self.held = set()

    def check_focus(self):
        if not self.keyboard.is_foreground():
            self.release_all()
            raise InputError("Game window lost focus")

    def wait_for_focus(self, timeout=30.0, clock=time.monotonic, sleep=time.sleep):
        """Wait without sending input. Focus loss during a task still aborts it."""
        if timeout <= 0:
            raise ValueError("Focus timeout must be positive")
        self.release_all()
        deadline = clock() + timeout
        while clock() < deadline:
            if self.keyboard.is_foreground():
                return
            sleep(0.05)
        raise InputError("Timed out waiting for the game window to gain focus")

    def apply(self, action):
        action = operator.index(action)
        if not 0 <= action < len(DIRECTIONS):
            raise ValueError("Action must be an integer from 0 to 8")
        self.check_focus()
        wanted = set(DIRECTIONS[action])
        if self.shoot:
            wanted.add("z")
        if self.slow:
            wanted.add("shift")
        try:
            for key in sorted(self.held - wanted):
                self.keyboard.key_up(key)
                self.held.remove(key)
            for key in sorted(wanted - self.held):
                self.check_focus()
                self.held.add(key)
                self.keyboard.key_down(key)
        except BaseException:
            self.release_all()
            raise

    def release_all(self):
        errors = []
        for key in sorted(self.held):
            try:
                self.keyboard.key_up(key)
                self.held.remove(key)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise InputError("Could not release all owned keys") from errors[0]

    def tap_confirm(self, sleep, duration=0.06):
        self.release_all()
        self.check_focus()
        self.held.add("z")
        try:
            self.keyboard.key_down("z")
            sleep(duration)
        finally:
            self.release_all()
