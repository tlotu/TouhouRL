"""Speculative planning for a four-frame replay boundary."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from .boundary_control import feasible_actions
from .replay_hrl import Choice, HOLD, PADDING, hold_outcomes, rebase, robust_routes
from .short_planner import Route, VECTORS
from .swept_collision import SweptCollision


def project(scene, position):
    circles = tuple(replace(c, x=c.x + HOLD * c.vx, y=c.y + HOLD * c.vy) for c in scene.circles)
    lasers = tuple(replace(l, on=max(0., l.on - HOLD), off=l.off - HOLD)
                   for l in scene.lasers if l.off > HOLD)
    return replace(scene, player=position, circles=circles, lasers=lasers)


def fallback_hold(scene, observation, belief, previous):
    """Rank only the next hold, using every modeled input timing."""
    padded = replace(scene, player_radius=scene.player_radius + PADDING)
    collision = SweptCollision(padded)
    actions = feasible_actions(observation, HOLD)
    candidates = [(action, moves) for action in actions for moves in sorted({
        state.advance(action, HOLD, delay)[0] for state in belief for delay in (0, 1)})]
    if not candidates:
        return None
    requests, moves = map(np.asarray, zip(*candidates))
    points = np.asarray(scene.player) + scene.speed * VECTORS[moves].cumsum(axis=1)
    starts = np.broadcast_to(scene.player, (len(points), 2))
    valid = np.ones(len(points), dtype=bool)
    gaps = np.full(len(points), collision.clearance_cap)
    for frame, ends in enumerate(points.transpose(1, 0, 2)):
        safe, clearance = collision.edges(starts, ends, frame, 1)
        valid &= safe
        gaps = np.minimum(gaps, clearance)
        starts = ends
    ttc = collision.stationary_ttc(points[:, -1], HOLD, 16)
    choices = []
    for action in actions:
        rows = requests == action
        if valid[rows].all():
            gap, tail = float(gaps[rows].min()), float(ttc[rows].min())
            point = points[np.flatnonzero(rows)[0], -1]
            choices.append((gap + .5 * tail, action == previous, -action,
                            Choice(Route((), tuple(point), (action,)), gap, tail, point)))
    return max(choices, key=lambda entry: entry[:3])[-1] if choices else None


class AsyncReplayPlanner:
    """A single outstanding job. Consume only at its target frame."""

    def __init__(self, worker):
        self.worker = worker
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def reset(self):
        if self.pending is not None:
            self.pending[1].cancel()
            # Keep running work bounded, but make its result ineligible.
            self.pending = (-1, *self.pending[1:])

    def submit(self, frame, scene, observation, belief, held):
        if self.pending is not None:
            if not self.pending[1].done():
                return False
            self.pending = None
        outcomes = hold_outcomes(scene, belief, held)
        if len(outcomes) != 1:
            return False
        position, states = next(iter(outcomes.items()))
        xmin, xmax, ymin, ymax = scene.bounds
        if not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax):
            return False
        def search():
            return self.worker.options(project(scene, position), rebase(observation, position), frozenset(states))

        future = self.pool.submit(search)
        self.pending = (frame + HOLD, future, position, frozenset(states))
        return True

    def consume(self, frame, scene, observation, belief):
        if self.pending is None:
            return None, 'empty'
        target, future, position, states = self.pending
        if frame != target:
            return None, 'stale'
        if not future.done():
            return None, 'late'
        self.pending = None
        if (max(abs(a - b) for a, b in zip(position, scene.player)) > .02
                or not belief.issubset(states)):
            return None, 'state_mismatch'
        choices, _, _ = future.result()
        if not choices:
            return None, 'no_proposal'
        padded = replace(scene, player_radius=scene.player_radius + PADDING)
        legal = feasible_actions(observation, HOLD)
        routes = robust_routes(padded, belief, (choices[0].route,), legal, SweptCollision(padded),
                               prune_failed=True)
        if not routes:
            return None, 'recheck_failed'
        route, gap, ttc = routes[0]
        return replace(choices[0], route=route, gap=gap, ttc=ttc), 'accepted'
