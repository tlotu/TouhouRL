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

    def __init__(self, keyboard, shoot=True, slow=True, *, profile_input=False, batch_multi_key=False):
        self.keyboard = keyboard
        self.shoot = shoot
        self.slow = slow
        self.held = set()
        self.profile_input = profile_input
        self.batch_multi_key = batch_multi_key

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

    def _wanted_keys(self, action):
        action = operator.index(action)
        if not 0 <= action < len(DIRECTIONS):
            raise ValueError("Action must be an integer from 0 to 8")
        wanted = set(DIRECTIONS[action])
        if self.shoot:
            wanted.add("z")
        if self.slow:
            wanted.add("shift")
        return wanted

    def apply(self, action):
        if self.profile_input:
            with self.keyboard.profile_input() as timing:
                self._apply(action)
            return timing
        self._apply(action)

    def _apply(self, action):
        wanted = self._wanted_keys(action)
        self.check_focus()
        try:
            releases = sorted(self.held - wanted)
            presses = sorted(wanted - self.held)
            if (self.batch_multi_key and len(releases) + len(presses) > 1
                    and {"shift", "z"}.isdisjoint(releases + presses)):
                self._send_batch(wanted, releases, presses)
                return
            for key in releases:
                self.keyboard.key_up(key)
                self.held.remove(key)
            for key in presses:
                self.check_focus()
                self.held.add(key)
                self.keyboard.key_down(key)
        except BaseException:
            self.release_all()
            raise

    def _send_batch(self, wanted, releases, presses):
        """Update ownership so the caller can release keys after partial submission."""
        changes = [(key, False) for key in releases] + [(key, True) for key in presses]
        try:
            sent = self.keyboard.send_batch(changes)
        except InputError:
            # The native sender checks focus before submitting any events.
            raise
        except BaseException:
            self.held.update(wanted)
            raise
        if sent != len(changes):
            if sent:
                self.held.update(wanted)
            raise InputError(f"SendInput accepted {sent} of {len(changes)} events")
        self.held = wanted

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
        self.tap_key("z", sleep, duration)

    def tap_key(self, key, sleep, duration=0.06):
        if key not in ("z", "down", "up"):
            raise ValueError("Unsupported menu key")
        self.release_all()
        self.check_focus()
        self.held.add(key)
        try:
            self.keyboard.key_down(key)
            sleep(duration)
        finally:
            self.release_all()
