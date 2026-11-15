"""Collision checks for the measured direction-key transition prefix."""
import numpy as np

from env.action_emulator import DIRECTIONS
from .direction_buffer import switch_states
from .short_planner import HOLD, VECTORS
from .swept_collision import SweptCollision


SWITCH_GUARD = {'contract': 'one-direction-event-per-frame-v1', 'start_delay_frames': [0, 1],
                'work_budget_ms': 4.75}
def settled_by(previous, requested, frame, previous_deadline):
    if previous == requested:
        return previous_deadline
    events = len(set(DIRECTIONS[previous]) ^ set(DIRECTIONS[requested]))
    return max(previous_deadline, frame + max(SWITCH_GUARD['start_delay_frames'])) + events


def check_switch_prefix(scene, previous, requested, expired):
    """Check the held move and any queued tail using the caller's padded scene."""
    events = switch_states(previous, requested)[1:]
    delays = SWITCH_GUARD['start_delay_frames'] if events else (0,)
    horizon = max(HOLD, len(events) + max(delays))
    paths = [[previous] * delay + list(events) + [requested] * (horizon - len(events) - delay)
             for delay in delays]
    starts = np.repeat([scene.player], len(paths), axis=0)
    velocity = scene.speed * VECTORS[np.asarray(paths)].transpose(1, 0, 2)
    model = SweptCollision(scene)
    for frame, moves in enumerate(velocity):
        if expired():
            return {'status': 'deadline', 'directions': paths, 'checked_frames': frame}
        ends = starts + moves
        safe, _ = model.edges(starts, ends, frame, 1)
        if not safe.all():
            return {'status': 'blocked', 'directions': paths, 'checked_frames': frame + 1}
        starts = ends
    return {'status': 'clear', 'directions': paths, 'checked_frames': horizon}
