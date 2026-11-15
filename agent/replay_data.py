"""Decision telemetry for offline replay and experiment reports."""
from dataclasses import asdict

import numpy as np

from .forward_model import _inside_field
from .short_planner import HOLD, TTC_HORIZON, VECTORS, route_points
from .swept_collision import SweptCollision


def first_move_forecast(scene):
    starts = np.broadcast_to(scene.player, (9, 2))
    ends = starts + VECTORS * scene.speed * HOLD
    model = SweptCollision(scene)
    safe, gaps = model.edges(starts, ends, 0, HOLD)
    inside = _inside_field(ends, scene.bounds)
    ttc = model.stationary_ttc(ends[safe], HOLD, TTC_HORIZON)
    times = dict(zip(np.flatnonzero(safe), map(float, ttc)))
    return ends, safe, [
        {'action': action, 'endpoint': ends[action].tolist(),
         'status': 'clear' if safe[action] else 'collision' if inside[action] else 'boundary',
         'clearance': float(gaps[action]) if inside[action] else None,
         'ttc_after_hold': times.get(action)}
        for action in range(9)
    ]


def decision_record(scene, frame, goal=None, *, result=None, action=None, probabilities=None):
    _, _, forecasts = first_move_forecast(scene)
    return {
        'frame': frame, 'player': list(scene.player), 'goal': asdict(goal) if goal else None,
        'action': action, 'probabilities': probabilities, 'directions': forecasts,
        'proposed': result.proposed.actions[0] if result and result.proposed else None,
        'stop_reason': result.stop_reason if result else None,
        'checked_edges': result.checked_edges if result else None,
        'layers': [asdict(layer) for layer in result.layers] if result else [],
        'route_points': route_points(scene, result.selected).tolist() if result and result.selected else [],
        'leaf_ttc': result.selected.ttc if result and result.selected and result.ttc_evaluated else None,
    }
