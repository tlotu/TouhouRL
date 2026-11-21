"""Swept AABB broad phase with continuous circle/capsule narrow phase."""
from dataclasses import dataclass

import numpy as np

from .forward_model import (
    ForwardModel, _circle_gaps, _edge_arrays, _inside_field, _point_segment, _segment_distance,
)


@dataclass
class CollisionCounts:
    boundary_pruned: int = 0
    potential_pairs: int = 0
    checked_pairs: int = 0


def _boxes(starts, ends, radius):
    padding = np.asarray(radius)[..., None] + 1e-9
    return np.minimum(starts, ends) - padding, np.maximum(starts, ends) + padding


def _overlap_pairs(starts, ends, boxes, *, cull_batch=False):
    low, high = np.minimum(starts, ends), np.maximum(starts, ends)
    box_low, box_high = boxes
    kept = None
    if cull_batch and len(low) >= 32 and len(box_low):
        # Every individual sweep is contained by this batch bounding box.
        envelope_low, envelope_high = low.min(axis=0), high.max(axis=0)
        intersects = ((box_low <= envelope_high) & (box_high >= envelope_low)).all(axis=1)
        if not intersects.all():
            kept = np.flatnonzero(intersects)
            box_low, box_high = box_low[kept], box_high[kept]
    edges, hazards = np.nonzero(
        (low[:, None, 0] <= box_high[:, 0]) & (high[:, None, 0] >= box_low[:, 0])
        & (low[:, None, 1] <= box_high[:, 1]) & (high[:, None, 1] >= box_low[:, 1])
    )
    return edges, hazards if kept is None else kept[hazards]


class SweptCollision(ForwardModel):
    """Match the reference validity and its clearance clipped at clearance_cap."""

    def __init__(self, scene, clearance_cap=24., *, cull_batches=False):
        super().__init__(scene)
        if not np.isfinite(clearance_cap) or clearance_cap <= 0:
            raise ValueError("Invalid clearance cap")
        self.clearance_cap = clearance_cap
        self.cull_batches = cull_batches
        self.counts = CollisionCounts()
        self._layers = {}
        self._lasers = np.array([(l.ax, l.ay, l.bx, l.by, l.width / 2 + scene.player_radius,
                                  l.on, l.off) for l in scene.lasers], dtype=float).reshape(-1, 7)
        self._radii = self.circles[:, 4] + scene.player_radius

    def _layer(self, frame, duration):
        key = frame, duration
        if key not in self._layers:
            center = self.circles[:, :2] + frame * self.circles[:, 2:4]
            end = self.circles[:, :2] + (frame + duration) * self.circles[:, 2:4]
            circle_boxes = _boxes(center, end, self._radii + self.clearance_cap)
            active = self._lasers[(self._lasers[:, 6] > self._lasers[:, 5])
                                  & (self._lasers[:, 6] > frame)
                                  & (self._lasers[:, 5] <= frame + duration)]
            laser_boxes = _boxes(active[:, :2], active[:, 2:4], active[:, 4] + self.clearance_cap)
            self._layers[key] = center, circle_boxes, active, laser_boxes
        return self._layers[key]

    def edges(self, starts, ends, frame, duration):
        starts, ends = _edge_arrays(starts, ends, frame, duration)
        indices = np.flatnonzero(_inside_field(starts, self.scene.bounds)
                                 & _inside_field(ends, self.scene.bounds))
        valid = np.zeros(len(starts), dtype=bool)
        gaps = np.full(len(starts), self.clearance_cap)
        self.counts.boundary_pruned += len(starts) - len(indices)
        if not len(indices):
            return valid, gaps
        starts, ends = starts[indices], ends[indices]
        moves = ends - starts
        center, circle_boxes, lasers, laser_boxes = self._layer(frame, duration)
        self.counts.potential_pairs += len(starts) * (len(center) + len(lasers))
        edge, circle = _overlap_pairs(starts, ends, circle_boxes, cull_batch=self.cull_batches)
        self.counts.checked_pairs += len(edge)
        if len(edge):
            velocity = moves[edge] / duration if duration else 0.
            values = _circle_gaps(center[circle] - starts[edge], self.circles[circle, 2:4] - velocity,
                                  self._radii[circle], duration)
            np.minimum.at(gaps, indices[edge], values)
        edge, laser = _overlap_pairs(starts, ends, laser_boxes, cull_batch=self.cull_batches)
        self.counts.checked_pairs += len(edge)
        if len(edge):
            beams = lasers[laser]
            if duration:
                lo = np.maximum(frame, beams[:, 5]) - frame
                hi = np.minimum(frame + duration, beams[:, 6]) - frame
                first = starts[edge] + moves[edge] * (lo / duration)[:, None]
                last = starts[edge] + moves[edge] * (hi / duration)[:, None]
                distance = _segment_distance(first, last, beams[:, :2], beams[:, 2:4])
            else:
                distance = _point_segment(starts[edge], beams[:, :2], beams[:, 2:4])
            np.minimum.at(gaps, indices[edge], distance - beams[:, 4])
        valid[indices] = gaps[indices] > 0
        return valid, gaps

    def stationary_ttc(self, points, frame, horizon, *, warnings_only=False):
        """Capped time to contact at rest, optionally from future laser activations."""
        points, _ = _edge_arrays(points, points, frame, horizon)
        center, circle_boxes, lasers, laser_boxes = self._layer(frame, horizon)
        ttc = np.full(len(points), horizon, dtype=float)
        if not warnings_only:
            row, circle = _overlap_pairs(points, points, circle_boxes)
            if len(row):
                delta = center[circle] - points[row]
                velocity = self.circles[circle, 2:4]
                a = np.sum(velocity ** 2, axis=1)
                b = np.sum(delta * velocity, axis=1)
                c = np.sum(delta ** 2, axis=1) - self._radii[circle] ** 2
                discriminant = b ** 2 - a * c
                hit = (a > 0) & (b < 0) & (discriminant >= 0)
                # Stable form of the smaller quadratic root for an approaching circle.
                denominator = np.where(hit, -b + np.sqrt(np.maximum(discriminant, 0.)), 1.)
                time = np.where(c <= 0, 0., np.where(hit, c / denominator, horizon))
                np.minimum.at(ttc, row, time)
        row, laser = _overlap_pairs(points, points, laser_boxes)
        if len(row):
            beams = lasers[laser]
            hit = _point_segment(points[row], beams[:, :2], beams[:, 2:4]) <= beams[:, 4]
            if warnings_only:
                hit &= beams[:, 5] > frame
            np.minimum.at(ttc, row[hit], np.maximum(0., beams[hit, 5] - frame))
        return ttc

    def snapshot_edges(self, starts, ends, frame):
        """Check spatial segments against hazard shapes at one instant."""
        center, circles, lasers, beams = self._layer(frame, 0)
        valid = np.ones(len(starts), dtype=bool)
        edge, circle = _overlap_pairs(starts, ends, circles)
        if len(edge):
            hit = _point_segment(center[circle], starts[edge], ends[edge]) <= self._radii[circle]
            valid[edge[hit]] = False
        edge, laser = _overlap_pairs(starts, ends, beams)
        if len(edge):
            rows = lasers[laser]
            hit = _segment_distance(starts[edge], ends[edge], rows[:, :2], rows[:, 2:4]) <= rows[:, 4]
            valid[edge[hit]] = False
        return valid


class UniqueSweeps(SweptCollision):
    """Evaluate identical edges once per batch and restore the original order."""

    def edges(self, starts, ends, frame, duration):
        if not (isinstance(starts, np.ndarray) and isinstance(ends, np.ndarray)
                and starts.dtype == ends.dtype == np.dtype(float)
                and starts.ndim == 2 and starts.shape[1] == 2
                and starts.shape == ends.shape and len(starts) >= 32):
            return super().edges(starts, ends, frame, duration)
        rows = np.column_stack((starts, ends))
        keys = rows.view(np.dtype((np.void, rows.strides[0]))).ravel()
        _, unique, inverse = np.unique(keys, return_index=True, return_inverse=True)
        valid, gaps = super().edges(starts[unique], ends[unique], frame, duration)
        return valid[inverse], gaps[inverse]
