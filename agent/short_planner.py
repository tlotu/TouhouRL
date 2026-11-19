"""Offline nine-direction beam planner with timed exit hints and stable actions."""
from dataclasses import dataclass, replace
import math
import time

import numpy as np

from env.action_emulator import DIRECTIONS
from .direction_buffer import DirectionBuffer
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


def exit_hint(scene, goal=None, *, expired=None, nearest_closure=False):
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
        if expired is not None and expired():
            return None
        mid = (lo + hi) / 2
        tail = origin + velocity[ray] * mid[:, None]
        intersects = _segment_distance(tail, ends[ray], first[beam], last[beam]) <= radius[beam]
        lo = np.where(intersects, mid, lo)
        hi = np.where(intersects, hi, mid)
    clear = np.ceil(hi).astype(int)
    slack = closing - clear - EXIT_DELAY
    selected = int(np.lexsort((slack, closing))[0] if nearest_closure else np.argmin(slack))
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
    buffer: DirectionBuffer | None = None
    motion: tuple[int, ...] = ()


@dataclass(frozen=True)
class SearchBudget:
    max_edges: int
    max_ms: float | None = None

    def __post_init__(self):
        if type(self.max_edges) is not int or self.max_edges < len(DIRECTIONS):
            raise ValueError('Edge budget must cover the first layer')
        if self.max_ms is not None and (not math.isfinite(self.max_ms) or self.max_ms <= 0):
            raise ValueError('Invalid time budget')


@dataclass(frozen=True)
class LayerCoverage:
    frame: int
    edges: int
    safe_edges: int
    before_pruning: tuple[int, ...]
    retained: tuple[int, ...]
    retained_routes: int


@dataclass
class Plan:
    proposed: Route | None
    selected: Route | None
    horizon: int
    hint: ExitHint | None
    counts: CollisionCounts
    layers: tuple[LayerCoverage, ...]
    stop_reason: str
    ttc_evaluated: bool = False
    candidates: tuple[Route, ...] = ()

    @property
    def smoothed(self):
        return self.selected is not self.proposed

    @property
    def checked_edges(self):
        return sum(layer.edges for layer in self.layers)

    def first_actions_at(self, frame):
        return next((layer.retained for layer in self.layers if layer.frame == frame), ())


def _score(route, goal, ttc_weight):
    distance = 0.
    if goal is not None and route.arrival is None:
        distance = math.hypot(route.point[0] - goal.x, route.point[1] - goal.y)
    clearance_weight = 0.5 if goal is not None else 1.
    value = clearance_weight * route.clearance + ttc_weight * route.ttc - TURN_WEIGHT * route.turn - distance
    return route.arrival is not None, -route.arrival if route.arrival is not None else 0, value


def _keep_diverse(routes, width, score, origin, cell_size):
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
    if cell_size is None:
        return retained + remaining[:width - len(retained)]

    def cell(route):
        return tuple(math.floor((p - root) / cell_size + 0.5) for p, root in zip(route.point, origin))

    occupied = {cell(route) for route in retained}
    duplicates = []
    for route in remaining:
        if len(retained) == width:
            break
        location = cell(route)
        if location in occupied:
            duplicates.append(route)
        else:
            occupied.add(location)
            retained.append(route)
    return retained + duplicates[:width - len(retained)]


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
         ttc_weight=TTC_WEIGHT, adaptive=False, smooth=True, budget=None,
         cell_size=None, clock=time.perf_counter, input_buffer: DirectionBuffer | None = None,
         switch_delay=0, deadline=None, leaf_reserve_ms=0., collision_model=None,
         horizon_override=None, first_action=None):
    """Search at a fresh four-frame boundary using known hazards.

    The absolute deadline uses the caller's clock. With it, search polls between
    small expansions and buffered motion frames. A partial layer can supply
    the deepest verified route. The optional TTC reserve stops expansion early
    so the last verified frontier can still be ranked. NumPy kernels are synchronous.
    """
    if type(previous_action) is not int or not 0 <= previous_action < len(DIRECTIONS):
        raise ValueError('Invalid previous action')
    if type(switch_delay) is not int or switch_delay not in (0, 1):
        raise ValueError('Switch delay must be zero or one frame')
    if width is None and adaptive:
        raise ValueError('Unpruned search is limited to the short horizon')
    if width is not None and (type(width) is not int or width < len(DIRECTIONS)):
        raise ValueError('Beam width must cover all first actions')
    if not math.isfinite(ttc_weight) or ttc_weight < 0:
        raise ValueError('Invalid TTC weight')
    if cell_size is not None and (not math.isfinite(cell_size) or cell_size <= 0):
        raise ValueError('Invalid spatial coverage cell size')
    if not math.isfinite(leaf_reserve_ms) or leaf_reserve_ms < 0:
        raise ValueError('Invalid TTC time reserve')
    if horizon_override is not None and (type(horizon_override) is not int
            or not SHORT_HORIZON <= horizon_override <= 104 or horizon_override % HOLD):
        raise ValueError('Invalid horizon override')
    if width is None and horizon_override is not None and horizon_override > SHORT_HORIZON:
        raise ValueError('Unpruned search is limited to the short horizon')
    if first_action is not None and not 0 <= first_action < len(DIRECTIONS):
        raise ValueError('Invalid first action')
    anytime = deadline is not None
    if budget and budget.max_ms is not None:
        local_deadline = clock() + budget.max_ms / 1000
        deadline = local_deadline if deadline is None else min(deadline, local_deadline)

    def time_expired():
        return deadline is not None and clock() >= deadline

    poll_expansion = time_expired if anytime else lambda: False
    hint = exit_hint(scene, goal, expired=poll_expansion) if adaptive and not poll_expansion() else None
    horizon = LONG_HORIZON if hint is not None and hint.slack_frames <= EXIT_SLACK else SHORT_HORIZON
    if horizon_override is not None:
        horizon = horizon_override
    if collision_model is not None and collision_model.scene is not scene:
        raise ValueError('Collision model must use the search scene')
    model = collision_model if collision_model is not None else SweptCollision(scene)
    counts_before = replace(model.counts)

    def search_counts():
        return CollisionCounts(model.counts.boundary_pruned - counts_before.boundary_pruned,
                               model.counts.potential_pairs - counts_before.potential_pairs,
                               model.counts.checked_pairs - counts_before.checked_pairs)

    if poll_expansion():
        return Plan(None, None, horizon, hint, search_counts(), (), 'time_limit')
    valid, _ = model.edges([scene.player], [scene.player], 0, 0)
    if not valid[0]:
        return Plan(None, None, horizon, hint, search_counts(), (), 'unsafe_start')
    reserve = 0.
    if anytime and ttc_weight and leaf_reserve_ms:
        reserve = min(leaf_reserve_ms / 1000, max(0., deadline - clock()) / 4)

    def expansion_expired():
        return deadline is not None and clock() >= deadline - reserve

    poll_expansion = expansion_expired if anytime else lambda: False
    arrival = 0 if goal is not None and goal.reached(scene.player, 0) else None
    routes = [Route((0, 0, 0, 0), scene.player, heading=previous_action, arrival=arrival, buffer=input_buffer)]
    step = scene.speed * HOLD
    origin = np.asarray(scene.player)

    def score(route):
        return _score(route, goal, ttc_weight)

    reached = checked_edges = 0
    layers = []
    stop_reason = 'horizon'
    for frame in range(0, horizon, HOLD):
        edge_count = len(routes) * len(DIRECTIONS)
        if expansion_expired():
            stop_reason = 'time_limit'
            break
        if budget and checked_edges + edge_count > budget.max_edges:
            stop_reason = 'edge_limit'
            break
        following = {}
        layer_edges = safe_edges = 0
        batch_size = 3 if anytime else len(routes)
        for start in range(0, len(routes), batch_size):
            if start and expansion_expired():
                stop_reason = 'time_limit'
                break
            batch = routes[start:start + batch_size]
            extensions = _expand(model, batch, origin, step, frame, goal, switch_delay, poll_expansion)
            layer_edges += len(batch) * len(DIRECTIONS)
            if extensions is None:
                stop_reason = 'time_limit'
                break
            safe_edges += len(extensions)
            for route in extensions:
                if frame == 0 and first_action is not None and route.actions[0] != first_action:
                    continue
                key = (route.offset, route.heading, route.actions[0], route.arrival is not None, route.buffer)
                if key not in following or score(route) > score(following[key]):
                    following[key] = route
        if poll_expansion():
            stop_reason = 'time_limit'
        checked_edges += layer_edges
        if not following:
            layers.append(LayerCoverage(frame + HOLD, layer_edges, 0, (), (), 0))
            if stop_reason != 'time_limit':
                stop_reason = 'no_safe_extension'
            break
        reached = frame + HOLD
        routes = list(following.values())
        before_pruning = tuple(sorted({route.actions[0] for route in routes}))
        # Preserve all final candidates until the common TTC ranking.
        if width is not None and reached != horizon:
            routes = _keep_diverse(routes, width, score, origin, cell_size)
        layers.append(LayerCoverage(reached, layer_edges, safe_edges, before_pruning,
                                    tuple(sorted({route.actions[0] for route in routes})), len(routes)))
        if stop_reason == 'time_limit':
            break
    if not reached:
        return Plan(None, None, horizon, hint, search_counts(), tuple(layers), stop_reason)
    ttc_evaluated = False
    if ttc_weight and (stop_reason != 'time_limit' or reserve) and (anytime or not time_expired()):
        times = []
        batch_size = 27 if anytime else len(routes)
        for start in range(0, len(routes), batch_size):
            batch_times = _leaf_ttc(model, routes[start:start + batch_size], reached, switch_delay,
                                    expired=time_expired if anytime else poll_expansion)
            if batch_times is None:
                stop_reason = 'time_limit'
                break
            times.extend(batch_times)
        if len(times) == len(routes):
            routes = [replace(route, ttc=float(ttc)) for route, ttc in zip(routes, times)]
            ttc_evaluated = True
    if width is not None and reached == horizon:
        routes = _keep_diverse(routes, width, score, origin, cell_size)
        layers[-1] = replace(layers[-1], retained=tuple(sorted({r.actions[0] for r in routes})),
                             retained_routes=len(routes))
    proposed, selected = _select(routes, score, previous_action, smooth)
    return Plan(proposed, selected, horizon, hint, search_counts(), tuple(layers), stop_reason,
                ttc_evaluated, tuple(routes))


def forced_beams(model, state, actions, delay, width):
    """Expand independent fixed-first-action beams in one collision batch per layer."""
    scene = model.scene
    if not actions or not model.edges([scene.player], [scene.player], 0, 0)[0][0]:
        return {}, 0
    origin = np.asarray(scene.player)
    roots = [Route((0, 0, 0, 0), scene.player, heading=state.requested, buffer=state)]
    checked = 0
    def score(route):
        return _score(route, None, 0.)
    for frame in range(0, SHORT_HORIZON, HOLD):
        # Report the same logical edge count as independent searches.
        checked += len(DIRECTIONS) * (len(actions) if frame == 0 else len(roots))
        following = {a: {} for a in actions}
        for r in _expand(model, roots, origin, scene.speed * HOLD, frame, None, delay, lambda: False):
            group = following.get(r.actions[0])
            if group is None:
                continue
            key = r.offset, r.heading, r.buffer
            if key not in group or score(r) > score(group[key]):
                group[key] = r
        beams = {a: _keep_diverse(group.values(), width, score, origin, None)
                 for a, group in following.items() if group}
        roots = [r for routes in beams.values() for r in routes]
        if not roots:
            break
    return beams, checked


def _expand(model, batch, origin, step, frame, goal, switch_delay, expired):
    starts = np.repeat(np.array([route.point for route in batch]), len(DIRECTIONS), axis=0)
    roots = np.array([route.offset for route in batch])
    buffered = batch[0].buffer is not None
    if buffered:
        transitions = [route.buffer.advance(action, HOLD, switch_delay)
                       for route in batch for action in range(len(DIRECTIONS))]
        motion = np.array([moves for moves, _ in transitions])
        trajectory = np.repeat(roots, len(DIRECTIONS), axis=0)[:, None, :] + OFFSETS[motion].cumsum(axis=1)
        points = origin + (step / HOLD) * (trajectory[..., :2] + trajectory[..., 2:] / math.sqrt(2))
        offsets, ends = trajectory[:, -1], points[:, -1]
        checked = _motion_edges(model, starts, points, frame, expired=expired)
        if checked is None:
            return None
        valid, clearance = checked
    else:
        offsets = (roots[:, None, :] + OFFSETS).reshape(-1, 4)
        ends = origin + step * (offsets[:, :2] + offsets[:, 2:] / math.sqrt(2))
        valid, clearance = model.edges(starts, ends, frame, HOLD)
    extensions = []
    for index in np.flatnonzero(valid):
        parent_index, action = divmod(int(index), len(DIRECTIONS))
        parent = batch[parent_index]
        point = tuple(map(float, ends[index]))
        arrival = parent.arrival
        if arrival is None and goal is not None and goal.reached(point, frame + HOLD):
            arrival = frame + HOLD
        moves, tail = transitions[index] if buffered else ((), None)
        extensions.append(Route(tuple(map(int, offsets[index])), point, parent.actions + (action,),
                                action or parent.heading, min(parent.clearance, float(clearance[index])),
                                parent.turn + float(ANGLES[parent.heading, action]), arrival,
                                buffer=tail, motion=parent.motion + moves))
    return extensions


def route_points(scene, route):
    """Return held endpoints for nominal motion or frame vertices for buffered motion."""
    if route.motion:
        offsets = np.vstack((np.zeros((1, 4), dtype=int), OFFSETS[list(route.motion)].cumsum(axis=0)))
        return scene.player + scene.speed * (offsets[:, :2] + offsets[:, 2:] / math.sqrt(2))
    actions = list(route.actions)
    offsets = np.vstack((np.zeros((1, 4), dtype=int), OFFSETS[actions].cumsum(axis=0)))
    return scene.player + scene.speed * HOLD * (offsets[:, :2] + offsets[:, 2:] / math.sqrt(2))


def _motion_edges(model, starts, points, frame, *, expired):
    valid = np.ones(len(starts), dtype=bool)
    clearance = np.full(len(starts), model.clearance_cap)
    for index in range(points.shape[1]):
        if expired():
            return None
        live = np.flatnonzero(valid)
        if not len(live):
            break
        ends = points[live, index]
        safe, gaps = model.edges(starts[live], ends, frame + index, 1)
        valid[live] = safe
        clearance[live] = np.minimum(clearance[live], gaps)
        starts = points[:, index]
    return valid, clearance


def _leaf_ttc(model, routes, frame, delay, *, expired):
    if expired():
        return None
    starts = np.array([route.point for route in routes])
    if routes[0].buffer is None:
        return model.stationary_ttc(starts, frame, TTC_HORIZON)
    moves = [route.buffer.advance(0, HOLD, delay)[0] for route in routes]
    points = starts[:, None, :] + model.scene.speed * VECTORS[np.array(moves)].cumsum(axis=1)
    times = np.full(len(routes), -1.)
    for i in range(HOLD):
        if expired():
            return None
        live = np.flatnonzero(times < 0)
        ends = points[live, i]
        safe, _ = model.edges(starts[live], ends, frame + i, 1)
        times[live[~safe]] = i  # Conservative lower bound within the contact frame.
        starts = points[:, i]
    live = np.flatnonzero(times < 0)
    if expired():
        return None
    times[live] = HOLD + model.stationary_ttc(starts[live], frame + HOLD, TTC_HORIZON - HOLD)
    return times
