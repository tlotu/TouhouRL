"""Offline nine-direction beam planner with timed exit hints and stable actions."""
from dataclasses import dataclass, replace
import math

import numpy as np

from env.action_emulator import DIRECTIONS
from .forward_model import _point_segment, _segment_distance
from .swept_collision import CollisionCounts, SweptCollision, _boxes, _overlap_pairs
from .timed_planner import TimedGoal


HOLD = 4
SHORT_HORIZON = 16
LONG_HORIZON = 60
TTC_HORIZON = 60
TURN_WEIGHT = 0.5
TTC_WEIGHT = 0.25
EXIT_SLACK = 12
EXIT_DELAY = 4
SMOOTH_TOLERANCE = 0.5

# A separate integer displacement count for each cardinal and diagonal axis
# avoids merging distinct positions just because they round to the same cell.
_moves = np.array([(int('right' in keys) - int('left' in keys),
                    int('down' in keys) - int('up' in keys)) for keys in DIRECTIONS])
_diagonal = np.count_nonzero(_moves, axis=1) == 2
OFFSETS = np.column_stack((_moves * (~_diagonal[:, None]), _moves * _diagonal[:, None]))
VECTORS = _moves / np.maximum(1., np.linalg.norm(_moves, axis=1))[:, None]
ANGLES = np.arccos(np.clip(VECTORS @ VECTORS.T, -1., 1.))
ANGLES[0, :] = ANGLES[:, 0] = 0.
np.fill_diagonal(ANGLES, 0.)


@dataclass(frozen=True)
class ExitHint:
    closing_frames: float
    clear_frames: int
    slack_frames: float
    source: str


def exit_hint(scene, goal=None):
    """Estimate closure on a goal ray or nearby escape rays from known warnings.

    Travel is nominal straight-line motion, capped at 32 frames. This heuristic
    detects threatened crossings and does not prove a topological bottleneck.
    """
    warnings = [laser for laser in scene.lasers if 0 < laser.on <= 96 and laser.off > laser.on]
    if not warnings:
        return None
    origin = np.asarray(scene.player)
    if goal is None:
        velocity = VECTORS[1:] * scene.speed
        duration = np.full(len(velocity), 32.)
    else:
        delta = np.array((goal.x, goal.y)) - origin
        distance = float(np.linalg.norm(delta))
        if distance == 0:
            return None
        velocity = (delta * (scene.speed / distance))[None, :]
        duration = np.array([min(32., distance / scene.speed)])
    xmin, xmax, ymin, ymax = scene.bounds
    wall = np.where(velocity > 0, (xmax, ymax), (xmin, ymin))
    time_to_wall = np.full_like(velocity, np.inf)
    np.divide(wall - origin, velocity, out=time_to_wall, where=velocity != 0)
    duration = np.maximum(0., np.minimum(duration, time_to_wall.min(axis=1)))
    ends = origin + velocity * duration[:, None]
    starts = np.broadcast_to(origin, ends.shape)
    first = np.array([(laser.ax, laser.ay) for laser in warnings])
    last = np.array([(laser.bx, laser.by) for laser in warnings])
    radius = np.array([laser.width / 2 + scene.player_radius for laser in warnings])
    ray, beam = _overlap_pairs(starts, ends, _boxes(first, last, radius))
    if not len(ray):
        return None
    closing = np.array([laser.on for laser in warnings])[beam]
    before_closure = origin + velocity[ray] * np.minimum(duration[ray], closing)[:, None]
    crosses = (_segment_distance(starts[ray], before_closure, first[beam], last[beam]) <= radius[beam])
    crosses &= _point_segment(ends[ray], first[beam], last[beam]) > radius[beam]
    ray, beam, closing = ray[crosses], beam[crosses], closing[crosses]
    if not len(ray):
        return None
    lo, hi = np.zeros(len(ray)), duration[ray].copy()
    # Find the last contact by testing whether the remaining ray intersects.
    for _ in range(8):
        mid = (lo + hi) / 2
        tail = origin + velocity[ray] * mid[:, None]
        intersects = _segment_distance(tail, ends[ray], first[beam], last[beam]) <= radius[beam]
        lo = np.where(intersects, mid, lo)
        hi = np.where(intersects, hi, mid)
    clear = np.ceil(hi).astype(int)
    slack = closing - clear - EXIT_DELAY
    selected = int(np.argmin(slack))
    return ExitHint(float(closing[selected]), int(clear[selected]), float(slack[selected]),
                    'local_rays' if goal is None else 'goal_ray')


@dataclass(frozen=True)
class Route:
    offset: tuple
    point: tuple
    actions: tuple = ()
    heading: int = 0
    clearance: float = 24.
    turn: float = 0.
    arrival: int | None = None
    ttc: float = 0.


@dataclass
class Plan:
    proposed: Route | None
    selected: Route | None
    horizon: int
    hint: ExitHint | None
    counts: CollisionCounts

    @property
    def smoothed(self):
        return self.selected is not self.proposed


def _score(route, goal, ttc_weight):
    distance = 0.
    if goal is not None and route.arrival is None:
        distance = math.hypot(route.point[0] - goal.x, route.point[1] - goal.y)
    clearance_weight = 0.5 if goal is not None else 1.
    value = clearance_weight * route.clearance + ttc_weight * route.ttc - TURN_WEIGHT * route.turn - distance
    return route.arrival is not None, -route.arrival if route.arrival is not None else 0, value


def _keep_diverse(routes, width, score):
    # For equal scores retain some sustained motions before spending the beam
    # on paths that differ only by where a wait was inserted.
    ranked = sorted(routes, key=lambda route: (*score(route), route.actions.count(route.actions[0])),
                    reverse=True)
    if len(ranked) <= width:
        return ranked
    retained, remaining = [], []
    quota = width // len(DIRECTIONS)
    counts = [0] * len(DIRECTIONS)
    for route in ranked:
        action = route.actions[0]
        if counts[action] < quota:
            retained.append(route)
            counts[action] += 1
        else:
            remaining.append(route)
    return retained + remaining[:width - len(retained)]


def _select(routes, score, previous_action, smooth):
    proposed = max(routes, key=lambda route: (*score(route), route.actions[0] == previous_action))
    action = proposed.actions[0]
    if not smooth or not previous_action or not action or action == previous_action:
        return proposed, proposed
    if ANGLES[previous_action, action] > math.pi / 4 + 1e-9:
        return proposed, proposed
    alternative = max((route for route in routes if route.actions[0] == previous_action),
                      key=score, default=None)
    if alternative is None:
        return proposed, proposed
    best_rank, alternative_rank = score(proposed), score(alternative)
    if (best_rank[:2] == alternative_rank[:2] and best_rank[2] - alternative_rank[2] <= SMOOTH_TOLERANCE
            and alternative.clearance >= proposed.clearance - SMOOTH_TOLERANCE
            and alternative.ttc >= proposed.ttc - 2):
        # This is a different route validated from the current state, not an
        # action substitution in the middle of the proposed route.
        return proposed, alternative
    return proposed, proposed


def plan(scene, goal: TimedGoal | None = None, *, previous_action=0, width=16,
         ttc_weight=TTC_WEIGHT, adaptive=False, smooth=True):
    """Search at a fresh four-frame action boundary using known hazards."""
    if type(previous_action) is not int or not 0 <= previous_action < len(DIRECTIONS):
        raise ValueError('Invalid previous action')
    if type(width) is not int or width < len(DIRECTIONS):
        raise ValueError('Beam width must cover all first actions')
    if not math.isfinite(ttc_weight) or ttc_weight < 0:
        raise ValueError('Invalid TTC weight')
    hint = exit_hint(scene, goal) if adaptive else None
    horizon = LONG_HORIZON if hint is not None and hint.slack_frames <= EXIT_SLACK else SHORT_HORIZON
    model = SweptCollision(scene)
    valid, _ = model.edges([scene.player], [scene.player], 0, 0)
    if not valid[0]:
        return Plan(None, None, horizon, hint, model.counts)
    arrival = 0 if goal is not None and goal.reached(scene.player, 0) else None
    routes = [Route((0, 0, 0, 0), scene.player, heading=previous_action, arrival=arrival)]
    step = scene.speed * HOLD
    origin = np.asarray(scene.player)

    def score(route):
        return _score(route, goal, ttc_weight)

    reached = 0
    for frame in range(0, horizon, HOLD):
        offsets = (np.array([route.offset for route in routes])[:, None, :] + OFFSETS).reshape(-1, 4)
        starts = np.repeat(np.array([route.point for route in routes]), len(DIRECTIONS), axis=0)
        ends = origin + step * (offsets[:, :2] + offsets[:, 2:] / math.sqrt(2))
        valid, clearance = model.edges(starts, ends, frame, HOLD)
        following = {}
        for index in np.flatnonzero(valid):
            parent_index, action = divmod(int(index), len(DIRECTIONS))
            parent = routes[parent_index]
            point = tuple(map(float, ends[index]))
            offset = tuple(map(int, offsets[index]))
            arrival = parent.arrival
            if arrival is None and goal is not None and goal.reached(point, frame + HOLD):
                arrival = frame + HOLD
            route = Route(offset, point, parent.actions + (action,), action or parent.heading,
                          min(parent.clearance, float(clearance[index])),
                          parent.turn + float(ANGLES[parent.heading, action]), arrival)
            key = (offset, route.heading, route.actions[0], arrival is not None)
            if key not in following or score(route) > score(following[key]):
                following[key] = route
        if not following:
            break
        reached = frame + HOLD
        routes = list(following.values())
        if reached == horizon and ttc_weight:
            times = model.stationary_ttc([route.point for route in routes], reached, TTC_HORIZON)
            routes = [replace(route, ttc=float(ttc)) for route, ttc in zip(routes, times)]
        routes = _keep_diverse(routes, width, score)
    if not reached:
        return Plan(None, None, horizon, hint, model.counts)
    if reached < horizon and ttc_weight:
        times = model.stationary_ttc([route.point for route in routes], reached, TTC_HORIZON)
        routes = [replace(route, ttc=float(ttc)) for route, ttc in zip(routes, times)]
    proposed, selected = _select(routes, score, previous_action, smooth)
    return Plan(proposed, selected, horizon, hint, model.counts)


def route_points(scene, route):
    """Reconstruct segment endpoints from the selected action sequence."""
    actions = list(route.actions)
    offsets = np.vstack((np.zeros((1, 4), dtype=int), OFFSETS[actions].cumsum(axis=0)))
    return scene.player + scene.speed * HOLD * (offsets[:, :2] + offsets[:, 2:] / math.sqrt(2))
