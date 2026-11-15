"""Forecast the free component containing the current player position."""
from dataclasses import replace

import numpy as np

from .replay_hrl import HOLD, PADDING
from .swept_collision import SweptCollision


def component_area(scene, horizon=400, spacing=16., *, start_frame=0):
    """Use four-connected swept edges, including barriers between free centers."""
    if (type(horizon) is not int or horizon < HOLD or horizon % HOLD or spacing <= 0
            or not 0 <= start_frame <= horizon or start_frame % HOLD):
        raise ValueError('Invalid connectivity grid or horizon')
    xmin, xmax, ymin, ymax = scene.bounds
    xs = np.arange(xmin, xmax + 1e-6, spacing)
    ys = np.arange(ymin, ymax + 1e-6, spacing)
    points = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    nodes = np.arange(len(points)).reshape(len(ys), len(xs))
    left = np.r_[nodes[:, :-1].ravel(), nodes[:-1].ravel()]
    right = np.r_[nodes[:, 1:].ravel(), nodes[1:].ravel()]
    near = np.flatnonzero(np.linalg.norm(points - scene.player, axis=1) <= spacing * 1.5)
    anchor = len(points)
    points = np.vstack((points, scene.player))
    left = np.r_[left, np.full(len(near), anchor)]
    right = np.r_[right, near]
    starts, ends = points[left], points[right]
    model = SweptCollision(replace(scene, player_radius=scene.player_radius + PADDING), clearance_cap=1.)
    samples = []
    for frame in range(start_frame, horizon + 1, HOLD):
        safe = model.snapshot_edges(starts, ends, frame)
        neighbors = [[] for _ in points]
        for a, b in zip(left[safe].tolist(), right[safe].tolist()):
            neighbors[a].append(b)
            neighbors[b].append(a)
        seen = {anchor}
        stack = [anchor]
        while stack:
            for other in neighbors[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        samples.append((frame, (len(seen) - 1) * spacing**2))
        model._layers.clear()
    return samples


def contraction(samples):
    """Detect a 40% loss versus the previous 32 frames and its settling time."""
    start = None
    for i, (frame, area) in enumerate(samples):
        reference = max(value for _, value in samples[max(0, i - 8):i + 1])
        if start is None:
            if reference > 0 and area <= .6 * reference:
                start = i
        if start is not None:
            if area == 0:
                return {'start': samples[start][0], 'end': frame, 'reason': 'zero'}
            if i - start >= 3:
                recent = [value for _, value in samples[i - 3:i + 1]]
                if max(recent) - min(recent) <= .1 * max(recent):
                    return {'start': samples[start][0], 'end': frame, 'reason': 'stable'}
    return {'start': samples[start][0], 'end': samples[-1][0], 'reason': 'capped'} if start is not None else None


def forecast_horizon(scene, minimum=400, maximum=800):
    """Scan at least 400 frames and extend unresolved contractions to their end."""
    samples = component_area(scene, horizon=minimum)
    event = contraction(samples)
    while event and event['reason'] == 'capped' and samples[-1][0] < maximum:
        start = samples[-1][0] + HOLD
        samples.extend(component_area(scene, min(maximum, start - HOLD + 100), start_frame=start))
        event = contraction(samples)
    return samples[-1][0], event, samples
