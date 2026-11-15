"""Direction-key request and consumption state for the measured input model."""
from dataclasses import dataclass
from functools import lru_cache

from env.action_emulator import DIRECTIONS


ACTIONS = {frozenset(keys): action for action, keys in enumerate(DIRECTIONS)}


@lru_cache(maxsize=81)
def switch_states(previous, requested):
    keys, wanted = set(DIRECTIONS[previous]), set(DIRECTIONS[requested])
    states = [previous]
    for key in sorted(keys - wanted):
        keys.remove(key)
        states.append(ACTIONS[frozenset(keys)])
    for key in sorted(wanted - keys):
        keys.add(key)
        states.append(ACTIONS[frozenset(keys)])
    return tuple(states)


@dataclass(frozen=True)
class DirectionBuffer:
    requested: int = 0
    applied: int = 0
    pending: tuple[int, ...] = ()

    @lru_cache(maxsize=2048)
    def advance(self, requested, frames, delay=0):
        """Append ordered events and consume one per frame, retaining the tail."""
        events = switch_states(self.requested, requested)[1:]
        queued = self.pending + events
        if events and not self.pending:
            queued = (self.applied,) * delay + queued
        moves = queued[:frames]
        applied = moves[-1] if moves else self.applied
        return moves + (applied,) * (frames - len(moves)), DirectionBuffer(requested, applied, queued[frames:])
