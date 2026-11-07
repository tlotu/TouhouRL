"""Offline constant-velocity circles and timed static capsules, in pixels/frames."""
from dataclasses import dataclass
import math

import numpy as np


MODEL_CONTRACT = "known-hazards-continuous-edges-v1"
DEFAULT_BOUNDS = (8., 376., 16., 432.)


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.vx, self.vy, self.radius)) or self.radius < 0:
            raise ValueError("Invalid circle")


@dataclass(frozen=True)
class TimedLaser:
    ax: float
    ay: float
    bx: float
    by: float
    width: float
    on: float
    off: float

    def __post_init__(self):
        values = (self.ax, self.ay, self.bx, self.by, self.width, self.on, self.off)
        if not all(math.isfinite(v) for v in values) or self.width < 0 or not 0 <= self.on <= self.off:
            raise ValueError("Invalid timed laser")


@dataclass(frozen=True)
class Scene:
    player: tuple
    circles: tuple = ()
    lasers: tuple = ()
    bounds: tuple = DEFAULT_BOUNDS
    player_radius: float = 1.25
    speed: float = 2.

    def __post_init__(self):
        if len(self.player) != 2 or len(self.bounds) != 4:
            raise ValueError("Invalid scene dimensions")
        if not all(math.isfinite(v) for v in (*self.player, *self.bounds, self.player_radius, self.speed)):
            raise ValueError("Non-finite scene")
        xmin, xmax, ymin, ymax = self.bounds
        if not xmin < xmax or not ymin < ymax or self.player_radius < 0 or self.speed <= 0:
            raise ValueError("Invalid scene settings")
        if not xmin - 1e-4 <= self.player[0] <= xmax + 1e-4 or not ymin - 1e-4 <= self.player[1] <= ymax + 1e-4:
            raise ValueError("Player outside field")
        if any(not isinstance(c, Circle) for c in self.circles) or any(not isinstance(l, TimedLaser) for l in self.lasers):
            raise ValueError("Unsupported hazard type")


def _point_segment(p, a, b):
    d = b - a
    projection = np.sum((p - a) * d, axis=-1) / np.maximum(np.sum(d * d, axis=-1), 1e-30)
    return np.linalg.norm(p - (a + np.clip(projection, 0, 1)[..., None] * d), axis=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_distance(a, b, c, d):
    """Distances between broadcastable pairs of segments, including crossings."""
    gap = np.minimum.reduce((_point_segment(a, c, d), _point_segment(b, c, d),
                             _point_segment(c, a, b), _point_segment(d, a, b)))
    u, v = b - a, d - c
    denominator = _cross(u, v)
    proper = np.abs(denominator) > 1e-12
    divisor = np.where(proper, denominator, 1.)
    s, t = _cross(c - a, v) / divisor, _cross(c - a, u) / divisor
    return np.where(proper & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), 0., gap)


def _edge_arrays(starts, ends, frame, duration):
    starts, ends = np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape != ends.shape:
        raise ValueError("Edges must have matching (n, 2) endpoints")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("Invalid edge coordinates")
    if not math.isfinite(frame) or frame < 0 or not math.isfinite(duration) or duration < 0:
        raise ValueError("Invalid edge interval")
    if duration == 0 and not np.array_equal(starts, ends):
        raise ValueError("Zero-duration edges cannot move")
    return starts, ends


def _inside_field(points, bounds):
    xmin, xmax, ymin, ymax = bounds
    return ((points[:, 0] >= xmin - 1e-4) & (points[:, 0] <= xmax + 1e-4)
            & (points[:, 1] >= ymin - 1e-4) & (points[:, 1] <= ymax + 1e-4))


def _circle_gaps(delta, relative, radius, duration):
    time = np.clip(-np.sum(delta * relative, axis=-1)
                   / np.maximum(np.sum(relative ** 2, axis=-1), 1e-30), 0, duration)
    return np.linalg.norm(delta + relative * time[..., None], axis=-1) - radius


class ForwardModel:
    """Exact segment tests within this hypothesis, not the native game model.

    Laser intervals use conservative closed endpoints when an edge overlaps an
    active interval. Future births, acceleration and moving axes are absent.
    """

    def __init__(self, scene):
        self.scene = scene
        self.circles = np.asarray([(c.x, c.y, c.vx, c.vy, c.radius) for c in scene.circles], dtype=np.float64).reshape(-1, 5)

    def edges(self, starts, ends, frame, duration):
        """Return physical/hazard validity and minimum hazard clearance per edge."""
        starts, ends = _edge_arrays(starts, ends, frame, duration)
        valid = _inside_field(starts, self.scene.bounds) & _inside_field(ends, self.scene.bounds)
        gaps = np.full(len(starts), np.inf)
        if len(self.circles):
            delta = self.circles[None, :, :2] + frame * self.circles[None, :, 2:4] - starts[:, None, :]
            velocity = (ends - starts) / duration if duration else np.zeros_like(starts)
            relative = self.circles[None, :, 2:4] - velocity[:, None, :]
            distances = _circle_gaps(delta, relative, self.circles[None, :, 4] + self.scene.player_radius, duration)
            gaps = np.minimum(gaps, distances.min(axis=1))
        for laser in self.scene.lasers:
            if laser.off <= laser.on or frame >= laser.off or frame + duration < laser.on:
                continue
            lo, hi = max(frame, laser.on), min(frame + duration, laser.off)
            if duration:
                a = starts + (ends - starts) * ((lo - frame) / duration)
                b = starts + (ends - starts) * ((hi - frame) / duration)
            else:
                a, b = starts, ends
            start, end = np.array((laser.ax, laser.ay)), np.array((laser.bx, laser.by))
            distance = _segment_distance(a, b, start, end) if duration else _point_segment(a, start, end)
            gaps = np.minimum(gaps, distance - laser.width / 2 - self.scene.player_radius)
        return valid & (gaps > 0), gaps


def scene_from_observation(observation):
    """Decode policy float32 geometry only; return explicit clipping diagnostics."""
    player = np.asarray(observation["player"], dtype=np.float32)
    bullets = np.asarray(observation["bullets"], dtype=np.float32)
    lasers = np.asarray(observation["lasers"], dtype=np.float32)
    if player.shape != (4,) or bullets.ndim != 2 or bullets.shape[1] != 7 or lasers.shape != (64, 12):
        raise ValueError("Unsupported observation layout")
    if not all(np.isfinite(a).all() for a in (player, bullets, lasers)):
        raise ValueError("Non-finite observation")
    if not np.isin(bullets[:, 6], (0, 1)).all() or not np.isin(lasers[:, 11], (0, 1)).all():
        raise ValueError("Invalid observation presence mask")
    point = ((float(player[0]) + 1.) * 192, (float(player[1]) + 1.) * 224)
    circles, beams = [], []
    present = bullets[bullets[:, 6] == 1]
    for row in present:
        if np.any(np.abs(row[:4]) > 1):
            raise ValueError("Bullet outside normalized observation range")
        circles.append(Circle(point[0] + float(row[0]) * 384, point[1] + float(row[1]) * 448,
                              float(row[2]) * 16, float(row[3]) * 16, float(row[4]) * 32))
    for row in lasers[lasers[:, 11] == 1]:
        phase, timer, warn, fire, fade = map(float, row[5:10])
        if any(v != int(v) or v < 0 for v in (phase, timer, warn, fire, fade)) or phase not in (0, 1, 2) or row[10] != 0:
            raise ValueError("Unsupported moving laser or phase")
        if row[4] < 0:
            raise ValueError("Negative laser width")
        if phase == 2:
            continue
        on = max(0., warn - timer) if phase == 0 else 0.
        off = on + fire if phase == 0 else max(0., fire - timer)
        beams.append(TimedLaser(*map(float, row[:5]), on, off))
    return Scene(point, tuple(circles), tuple(beams)), {
        "encoded_circles": len(circles), "encoded_lasers": int(lasers[:, 11].sum()),
        "saturated_circle_rows": int(np.any(np.abs(present[:, :4]) >= 1, axis=1).sum()),
        "future_births_modeled": False, "native_collision_guarantee": False,
        "geometry_source": "float32 policy observation; omitted objects are unavailable"}
