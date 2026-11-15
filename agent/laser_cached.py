"""Budgeted laser planning with rechecked action suffixes."""
from dataclasses import replace

import numpy as np

from .boundary_control import feasible_actions
from .forward_model import scene_from_observation
from .input_switch import SWITCH_GUARD, check_switch_prefix, settled_by
from .laser_evaluation import LaserEvaluationCommitment
from .short_planner import HOLD, LONG_HORIZON, SHORT_HORIZON, TTC_HORIZON, VECTORS, SearchBudget, plan
from .swept_collision import SweptCollision


CACHE_CONFIG = {'contract': 'laser-cached-ttc-v1', 'clearance_px': 2., 'current_ttc_below': 20.,
                'alternative_ttc_above': 40., 'max_edges': 900, 'work_budget_ms': 3.5, 'refresh_budget_ms': 4.,
                'max_input_age_ms': 14., 'max_cache_age_frames': 16, 'beam_width': 9}


def recheck_routes(scene, routes, action, remaining, expired):
    """Certify cached alternatives and continuations to the same leaf time."""
    steps = np.asarray(routes, dtype=int)
    alternatives = steps[steps[:, 0] != action]
    if not len(alternatives):
        return None
    steps = np.vstack((steps, alternatives))
    continuation = np.arange(len(steps)) < len(routes)
    moves = VECTORS[steps] * scene.speed
    first = moves[:, 0].copy()
    first[continuation] = VECTORS[action] * scene.speed
    if remaining == HOLD:
        moves[:, 0] = first
    # On a held step, the continuation may switch as soon as the old hold expires.
    intervals = [(0, remaining, first)]
    if remaining < HOLD:
        intervals.append((remaining, HOLD - remaining, moves[:, 0]))
    intervals.extend((i * HOLD, HOLD, moves[:, i]) for i in range(1, steps.shape[1]))
    model = SweptCollision(scene)
    starts = np.repeat([scene.player], len(steps), axis=0)
    gaps = np.full(len(steps), model.clearance_cap)
    valid = np.ones(len(steps), dtype=bool)
    baseline_safe = None
    for frame, duration, velocity in intervals:
        if expired():
            return None
        indices = np.flatnonzero(valid)
        ends = starts[indices] + duration * velocity[indices]
        safe, local = model.edges(starts[indices], ends, frame, duration)
        valid[indices] = safe
        gaps[indices] = np.minimum(gaps[indices], local)
        starts[indices] = ends
        if frame == 0:
            baseline_safe = bool(valid[0])
        if not np.any(valid & ~continuation):
            return None
    if expired():
        return None
    horizon = steps.shape[1] * HOLD
    indices = np.flatnonzero(valid)
    ttcs = model.stationary_ttc(starts[indices], horizon, TTC_HORIZON)
    old = [float(ttc) for i, ttc in zip(indices, ttcs) if continuation[i]]
    new = max(((float(ttc), float(gaps[i]), int(steps[i, 0])) for i, ttc in zip(indices, ttcs)
               if not continuation[i]), default=None)
    return {'suggested_action': new[2], 'alternative_ttc': new[0], 'baseline_ttc': max(old, default=None),
            'baseline_first_hold_safe': baseline_safe, 'prefix_frames': horizon,
            'clearance_px': new[1] + CACHE_CONFIG['clearance_px'], 'source': 'cache'}


def compare_search(result, action):
    routes = result.candidates
    alternatives = [route for route in routes if route.actions[0] != action]
    if not alternatives or len(alternatives[0].actions) * HOLD < SHORT_HORIZON:
        return None
    first = result.layers[0]
    ranked = max(alternatives, key=lambda route: (route.ttc, route.clearance, -route.turn))
    current_ttc = max((route.ttc for route in routes if route.actions[0] == action), default=None)
    return {'suggested_action': ranked.actions[0],
            'alternative_ttc': ranked.ttc if result.ttc_evaluated else None,
            'baseline_ttc': current_ttc if result.ttc_evaluated else None,
            'baseline_first_hold_safe': action in first.before_pruning,
            'prefix_frames': len(ranked.actions) * HOLD,
            'clearance_px': ranked.clearance + CACHE_CONFIG['clearance_px'], 'source': 'search'}


def intervention_reason(candidate):
    if candidate is None:
        return None
    if candidate['baseline_first_hold_safe'] is False:
        return 'baseline_clearance'
    old, new = candidate['baseline_ttc'], candidate['alternative_ttc']
    if (old is not None and new is not None and old < CACHE_CONFIG['current_ttc_below']
            and new > CACHE_CONFIG['alternative_ttc_above']):
        return 'leaf_ttc'
    return None


class CachedLaserCommitment(LaserEvaluationCommitment):
    def __init__(self, *args, sample_status, read_frame, intervene=False, read_direction=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hold_frames != HOLD:
            raise ValueError('Cached planner requires four-frame holds')
        self.sample_status, self.read_frame, self.intervene = sample_status, read_frame, intervene
        self.read_direction = read_direction
        self.switch_action, self.switch_settles = 0, HOLD
        self.cached_routes = ()
        self.cached_frame = self.cached_horizon = None

    def annotate(self, row):
        if row['event'] == 'reset':
            self.cached_routes = ()
            self.cached_frame = self.cached_horizon = None
            self.switch_action, self.switch_settles = 0, HOLD
        if self.read_direction is not None and row['event'] == 'transition':
            info = row['info']
            frame = info['elapsed_frames'] - info['delta_frames']
            self.switch_settles = settled_by(self.switch_action, row['action'], frame, self.switch_settles)
            self.switch_action = row['action']
        return super().annotate(row)

    def __call__(self, obs):
        previous = self.held_action
        action = super().__call__(obs)
        frame = self.elapsed_frames()
        sample_frame, received = self.sample_status()
        held = self.last['commitment_active']
        remaining = max(2, HOLD - self.last['hold_age_frames']) if held else HOLD
        started = self.clock()
        age_deadline = received + CACHE_CONFIG['max_input_age_ms'] / 1000
        deadline = min(age_deadline, started + CACHE_CONFIG['work_budget_ms'] / 1000)
        expired = lambda: self.clock() >= deadline
        candidate, stop, attempted, worked = None, 'no_cache', False, False
        prefix = 0
        horizon = None
        if (not held or self.cached_routes) and not expired():
            worked = True
            scene, _ = scene_from_observation(obs)
            scene = replace(scene, player_radius=scene.player_radius + CACHE_CONFIG['clearance_px'])
            age = frame - self.cached_frame if self.cached_routes else None
            warning = any(0 < laser.on <= 32 and laser.off > laser.on for laser in scene.lasers)
            use_cache = (age is not None and age < CACHE_CONFIG['max_cache_age_frames']
                         and (held or not warning or self.cached_horizon == LONG_HORIZON))
            if use_cache:
                shift = age // HOLD
                shifted = [route[shift:] + (0,) * shift for route in self.cached_routes]
                candidate = recheck_routes(scene, shifted, action, remaining, expired)
                stop = 'cache_checked' if candidate else 'cache_unavailable'
                horizon = self.cached_horizon
            needs_refresh = (candidate is not None and candidate['baseline_ttc'] is not None
                             and candidate['baseline_ttc'] < CACHE_CONFIG['current_ttc_below']
                             and candidate['alternative_ttc'] <= CACHE_CONFIG['alternative_ttc_above'])
            search_deadline = (min(age_deadline, started + CACHE_CONFIG['refresh_budget_ms'] / 1000)
                               if needs_refresh else deadline)
            time_left = search_deadline - self.clock()
            if not held and (candidate is None or needs_refresh) and time_left > 0:
                attempted = True
                result = plan(scene, previous_action=0 if previous is None else previous, adaptive=True,
                              width=CACHE_CONFIG['beam_width'],
                              budget=SearchBudget(CACHE_CONFIG['max_edges'], time_left * 1000), clock=self.clock)
                searched = compare_search(result, action)
                if candidate is None or intervention_reason(searched):
                    candidate = searched
                stop, horizon = result.stop_reason, result.horizon
                prefix = len(result.selected.actions) * HOLD if result.selected else 0
                if prefix >= SHORT_HORIZON or not use_cache:
                    self.cached_routes = tuple(route.actions for route in result.candidates) if prefix >= SHORT_HORIZON else ()
                    self.cached_frame, self.cached_horizon = frame, horizon
        reason = intervention_reason(candidate)
        same_frame = None
        can_apply = False
        allowed = None
        switch_check = None
        if reason is not None:
            allowed = feasible_actions(obs, HOLD)
            if self.read_direction is not None:
                switch_deadline = min(age_deadline, started + SWITCH_GUARD['work_budget_ms'] / 1000)
                switch_check = self._check_switch(scene, frame, sample_frame, candidate['suggested_action'],
                                                  lambda: self.clock() >= switch_deadline)
            same_frame = self.read_frame() == sample_frame
            can_apply = (same_frame and self.clock() < age_deadline
                         and (switch_check is None or switch_check['status'] == 'clear')
                         and candidate['suggested_action'] in allowed)
        telemetry = {'baseline_action': action, 'baseline_hold_active': held, 'candidate': candidate,
                     'search_attempted': attempted, 'stop_reason': stop, 'horizon_frames': horizon,
                     'search_prefix_frames': prefix, 'reason': reason, 'eligible': can_apply,
                     'sample_frame': sample_frame, 'same_frame': same_frame, 'worked': worked,
                     'work_seconds': self.clock() - started,
                     'applied': self.intervene and can_apply}
        if self.read_direction is not None:
            telemetry['switch_check'] = switch_check
        self.last['planner_cached'] = telemetry
        if telemetry['applied']:
            action = candidate['suggested_action']
            override = self.last['raw_student_action'] != action
            self.total_overrides += int(override) - int(self.last['action_overridden'])
            self.held_action, self.since = action, frame
            self.last.update(committed_action=action, commitment_active=False, hold_age_frames=0,
                             action_overridden=override, boundary_execution_horizon_frames=HOLD,
                             boundary_execution_allowed_actions=allowed)
        return action

    def _check_switch(self, scene, frame, sample_frame, suggested, expired):
        if frame < self.switch_settles:
            return {'status': 'pending_keys'}
        if expired():
            return {'status': 'deadline'}
        if self.read_direction(sample_frame) != self.switch_action:
            return {'status': 'unconfirmed_direction'}
        return check_switch_prefix(scene, self.switch_action, suggested, expired)
