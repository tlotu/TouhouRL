"""Bounded retries for an explicitly selected spell-practice task."""
import time


class ResetError(RuntimeError):
    pass


class ResetController:
    def __init__(self, reader, controls, spell_retry=False, timeout=10.0,
                 clock=time.monotonic, sleep=time.sleep, menu_settle=0.35,
                 confirm_interval=0.75, max_confirms=3):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.reader, self.controls = reader, controls
        self.spell_retry, self.timeout = spell_retry, timeout
        self.clock, self.sleep = clock, sleep
        if menu_settle < 0 or confirm_interval <= 0 or max_confirms < 1:
            raise ValueError("Invalid retry timing or confirmation limit")
        self.menu_settle = menu_settle
        self.confirm_interval = confirm_interval
        self.max_confirms = max_confirms

    def reset(self, first=False):
        self.controls.release_all()
        deadline = self.clock() + self.timeout
        confirmed = False
        initial_frame = None
        menu_since = None
        next_confirm = 0.0
        confirms = 0
        left_menu = False
        sample = None
        menu_signature = None
        menu_outcome = None
        navigation_presses = 0
        next_navigation = 0.0
        while self.clock() < deadline:
            self.controls.check_focus()
            sample = self.reader.snapshot()
            menu = sample.result_menu
            if first and sample.ready:
                if initial_frame is None:
                    initial_frame = sample.frame
                elif sample.frame != initial_frame:
                    return sample
            if menu and menu.active:
                if not self.spell_retry:
                    raise ResetError("Spell-result retry requires explicit spell-practice mode")
                if menu.outcome is not None:
                    if menu_outcome is not None and menu_outcome != menu.outcome:
                        raise ResetError("Result outcome changed during menu navigation")
                    menu_outcome = menu.outcome
                signature = (menu.phase, menu.selection, menu.outcome_flag)
                if signature != menu_signature:
                    menu_signature, menu_since = signature, self.clock()
                settled = self.clock() - menu_since >= self.menu_settle
                if (not left_menu and menu.phase == 1 and settled
                        and self.clock() >= next_navigation):
                    if menu.selection != 2:
                        if navigation_presses >= 6:
                            raise ResetError("Retry selection did not respond to navigation")
                        self.controls.tap_key("down", self.sleep)
                        navigation_presses += 1
                        next_navigation = self.clock() + 0.15
                    elif confirms < self.max_confirms and self.clock() >= next_confirm:
                        self.controls.tap_confirm(self.sleep)
                        confirmed = True
                        confirms += 1
                        initial_frame = sample.frame
                        next_confirm = self.clock() + self.confirm_interval
            elif sample.retry and menu is None:
                if not self.spell_retry:
                    raise ResetError("Enter spell practice manually or explicitly enable spell retry")
                if menu_since is None:
                    menu_since = self.clock()
                # The failure flag can precede the interactive menu. Retry only
                # while the same menu remains visible, never after scene loading.
                if (not left_menu and confirms < self.max_confirms
                        and self.clock() - menu_since >= self.menu_settle
                        and self.clock() >= next_confirm):
                    self.controls.tap_confirm(self.sleep)
                    confirmed = True
                    confirms += 1
                    initial_frame = sample.frame
                    next_confirm = self.clock() + self.confirm_interval
            elif confirmed and sample.ready and sample.frame != initial_frame:
                return sample
            elif not first and not confirmed and sample.ready:
                raise ResetError("Task is still running; a new episode was not started")
            elif sample.scene in (1, 6, 7, 10):
                raise ResetError("Manual task selection is required from this screen")
            if not sample.retry and not (menu and menu.active):
                menu_since = None
                if confirmed:
                    left_menu = True
            self.sleep(0.01)
        details = (f"scene={sample.scene}, wanted={sample.wanted_scene}, "
                   f"retry={sample.retry}, dying={sample.dying}, "
                   f"player_state={sample.player_state}, frame={sample.frame}" if sample else "no sample")
        raise ResetError(f"Timed out waiting for a playable spell-practice task; "
                         f"confirms={confirms}, left_menu={left_menu}, {details}, "
                         f"result_menu={sample.result_menu if sample else None}")
