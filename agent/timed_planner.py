"""Offline held cardinal-action reference search and bounded-width search."""
from dataclasses import dataclass
import math
import time

import numpy as np

from .forward_model import ForwardModel


PLANNER_CONTRACT = "timed-cardinal-held-reachability-v1"
# Both searches share an exact lattice and a strict subset of native actions.
ACTION_IDS = (0, 1, 3, 5, 7)
UNIT_VECTORS = np.array(((0., 0.), (0., -1.), (1., 0.), (0., 1.), (-1., 0.)))


@dataclass(frozen=True)
class TimedGoal:
    x: float
    y: float
    earliest: int
    deadline: int
    radius: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.radius)) or self.radius < 0:
            raise ValueError("Invalid goal geometry")
        if type(self.earliest) is not int or type(self.deadline) is not int or not 0 <= self.earliest <= self.deadline:
            raise ValueError("Invalid goal time window")

    def reached(self, point, frame):
        return self.earliest <= frame <= self.deadline and math.hypot(point[0] - self.x, point[1] - self.y) <= self.radius


@dataclass(frozen=True)
class Path:
    point: tuple
    actions: tuple = ()
    clearance: float = math.inf


@dataclass
class SearchResult:
    method: str
    layers: list
    goal_path: object
    hold_frames: int
    horizon: int
    edges_tested: int
    pruned_states: int
    seconds: float

    @property
    def reached_frames(self):
        return (len(self.layers) - 1) * self.hold_frames

    @property
    def frontier(self):
        return list(self.layers[-1].values())


def search(scene, horizon=64, hold_frames=4, beam_width=None, goal=None, position_score=None):
    """Enumerate safe lattice states or prune them to beam_width per time layer.

    Reference completeness applies only to the five declared held actions and
    supplied known-hazard model. A goal path proves safe arrival only; it need
    not remain viable afterwards. Layers contain reachable, not all safe states.
    """
    if type(hold_frames) is not int or not 1 <= hold_frames <= 16:
        raise ValueError("Invalid action hold")
    if type(horizon) is not int or horizon <= 0 or horizon > 256 or horizon % hold_frames:
        raise ValueError("Horizon must be a positive hold multiple, at most 256")
    if beam_width is not None and (type(beam_width) is not int or beam_width < len(ACTION_IDS)):
        raise ValueError("Beam width must cover each first action")
    started = time.perf_counter()
    model = ForwardModel(scene)
    valid, _ = model.edges([scene.player], [scene.player], 0, 0)
    layers = [{(0, 0): Path(scene.player)}] if valid[0] else [{}]
    goal_path = layers[0].get((0, 0)) if goal and goal.reached(scene.player, 0) else None
    tested = pruned = 0
    increments = UNIT_VECTORS.astype(int)
    step = scene.speed * hold_frames

    def score(path):
        extra = float(position_score(path.point)) if position_score else 0.
        distance = -math.hypot(path.point[0] - goal.x, path.point[1] - goal.y) if goal else 0.
        return extra, distance, min(24., path.clearance)

    for frame in range(0, horizon, hold_frames):
        previous = list(layers[-1].items())
        if not previous:
            break
        starts = np.repeat(np.array([p.point for _, p in previous]), len(ACTION_IDS), axis=0)
        ends = starts + np.tile(UNIT_VECTORS * step, (len(previous), 1))
        valid, gaps = model.edges(starts, ends, frame, hold_frames)
        tested += len(starts)
        following = {}
        for index in np.flatnonzero(valid):
            parent_index, direction = divmod(int(index), len(ACTION_IDS))
            key, parent = previous[parent_index]
            newkey = (key[0] + int(increments[direction, 0]), key[1] + int(increments[direction, 1]))
            # Derive coordinates from lattice indices; avoid cumulative drift.
            point = (scene.player[0] + newkey[0] * step, scene.player[1] + newkey[1] * step)
            path = Path(point, parent.actions + (ACTION_IDS[direction],), min(parent.clearance, float(gaps[index])))
            if newkey not in following or path.clearance > following[newkey].clearance:
                following[newkey] = path
        if goal and goal_path is None:
            arrivals = [p for p in following.values() if goal.reached(p.point, frame + hold_frames)]
            if arrivals:
                goal_path = max(arrivals, key=score)
        if beam_width and len(following) > beam_width:
            ordered = sorted(following, key=lambda k: score(following[k]), reverse=True)
            kept = []
            quota = beam_width // len(ACTION_IDS)
            # Preserve multiple root directions before filling remaining slots.
            for action in ACTION_IDS:
                group = [k for k in ordered if following[k].actions[0] == action]
                kept.extend(group[:quota])
            chosen = set(kept)
            kept.extend(k for k in ordered if k not in chosen and len(kept) < beam_width)
            pruned += len(following) - len(kept)
            following = {k: following[k] for k in kept}
        if not following:
            break
        layers.append(following)
    return SearchResult("reference" if beam_width is None else "beam", layers, goal_path,
                        hold_frames, horizon, tested, pruned, time.perf_counter() - started)


def path_points(scene, path, hold_frames=4):
    """Expand a checked path to held-action endpoints."""
    points = [scene.player]
    velocities = dict(zip(ACTION_IDS, UNIT_VECTORS))
    for action in path.actions:
        vx, vy = velocities[action] * scene.speed * hold_frames
        points.append((points[-1][0] + float(vx), points[-1][1] + float(vy)))
    return points


def replay_path(scene, path, hold_frames=4):
    """Independently recheck each returned segment without invoking a policy."""
    model = ForwardModel(scene)
    points = path_points(scene, path, hold_frames)
    valid, _ = model.edges([points[0]], [points[0]], 0, 0)
    if not valid[0]:
        return False
    for index, (a, b) in enumerate(zip(points, points[1:])):
        valid, _ = model.edges([a], [b], index * hold_frames, hold_frames)
        if not valid[0]:
            return False
    return np.allclose(points[-1], path.point, rtol=0, atol=1e-9)
