"""Evaluate a bounded planner alongside the existing laser controller."""
from dataclasses import asdict, replace
from itertools import groupby

import numpy as np

from .boundary_control import feasible_actions
from .direction_buffer import DirectionBuffer
from .forward_model import scene_from_observation
from .laser_evaluation import LaserEvaluationCommitment
from .short_planner import HOLD, SHORT_HORIZON, TTC_HORIZON, VECTORS, SearchBudget, plan
from .swept_collision import SweptCollision


SHADOW_CONFIG = {
    'contract': 'laser-planner-shadow-v1',
    'adaptive': True,
    'budget': asdict(SearchBudget(900, 8.)),
    'fresh_decisions_only': True,
}
BUFFERED_TIMING_CONFIG = {**SHADOW_CONFIG, 'contract': 'laser-buffered-timing-shadow-v1',
                          'width': 9, 'clearance_px': 2., 'start_delays': [0, 1],
                          'initial_queue': 'assume_previous_request_settled'}
BUFFERED_REUSE_CONFIG = {**BUFFERED_TIMING_CONFIG, 'contract': 'laser-buffered-reuse-shadow-v2',
                         'fresh_decisions_only': False, 'initial_queue': 'track_baseline_requests',
                         'minimum_fresh_reuse_frames': SHORT_HORIZON,
                         'capture_to_planner_deadline_ms': 15.6}
BUFFERED_LEAF_CONFIG = {**BUFFERED_REUSE_CONFIG, 'contract': 'laser-buffered-leaf-shadow-v3',
                        'leaf_reserve_ms': .9, 'cache_warning_lookahead_frames': TTC_HORIZON}


class LaserShadowCommitment(LaserEvaluationCommitment):
    """Log a proposal at each new hold while returning the baseline action."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hold_frames != HOLD:
            raise ValueError('Planner shadow requires four-frame holds')
        self.budget = SearchBudget(**SHADOW_CONFIG['budget'])

    def __call__(self, obs):
        previous = self.held_action
        action = super().__call__(obs)
        if self.last['commitment_active']:
            return action

        started = self.clock()
        scene, metadata = scene_from_observation(obs)
        planner_started = self.clock()
        result = self._plan(scene, 0 if previous is None else previous)
        finished = self.clock()
        route = result.selected
        suggestion = route.actions[0] if route else None
        first_layer = next((layer for layer in result.layers if layer.frame == HOLD), None)
        baseline_safe = False if result.stop_reason == 'unsafe_start' else None
        if first_layer is not None:
            baseline_safe = action in first_layer.before_pruning
        self.last['planner_shadow'] = {
            'baseline_action': action,
            'suggested_action': suggestion,
            'previous_action': previous,
            'baseline_first_hold_safe': baseline_safe,
            'horizon_frames': result.horizon,
            'checked_prefix_frames': len(route.actions) * HOLD if route else 0,
            'route_actions': list(route.actions) if route else [],
            'stop_reason': result.stop_reason,
            'checked_edges': result.checked_edges,
            'exit_hint': asdict(result.hint) if result.hint else None,
            'leaf_ttc_frames': route.ttc if route and result.ttc_evaluated else None,
            'encoded_circles': metadata['encoded_circles'],
            'saturated_circle_rows': metadata['saturated_circle_rows'],
            'planner_seconds': finished - planner_started,
            'shadow_seconds': finished - started,
        }
        return action

    def _plan(self, scene, previous):
        return plan(scene, previous_action=previous, adaptive=True, budget=self.budget, clock=self.clock)


class LaserBufferedTimingCommitment(LaserShadowCommitment):
    """Measure the buffered planner with hypothetical settled keys and baseline input."""

    def _plan(self, scene, previous):
        delay = BUFFERED_TIMING_CONFIG['start_delays'][(self.elapsed_frames() // HOLD) % 2]
        padded = replace(scene, player_radius=scene.player_radius + BUFFERED_TIMING_CONFIG['clearance_px'])
        return plan(padded, previous_action=previous, width=BUFFERED_TIMING_CONFIG['width'],
                    adaptive=True, budget=self.budget, clock=self.clock,
                    input_buffer=DirectionBuffer(previous, previous), switch_delay=delay)


class LaserBufferedReuseCommitment(LaserEvaluationCommitment):
    """Rebase a buffered action suffix on each observation in shadow mode."""

    def __init__(self, *args, capture_started=None, planner_config=BUFFERED_LEAF_CONFIG, **kwargs):
        super().__init__(*args, **kwargs)
        if self.hold_frames != HOLD:
            raise ValueError('Buffered planner requires four-frame holds')
        self.capture_started = capture_started
        self.config = planner_config
        self.budget = SearchBudget(self.config['budget']['max_edges'])
        self.buffer = DirectionBuffer()
        self.cached_actions = ()
        self.cached_frame = None
        self.cached_horizon = None

    @staticmethod
    def _delay(frame):
        return BUFFERED_REUSE_CONFIG['start_delays'][(frame // HOLD) % 2]

    def annotate(self, row):
        if row['event'] == 'reset':
            self.buffer = DirectionBuffer()
            self.cached_actions = ()
            self.cached_frame = self.cached_horizon = None
        if row['event'] == 'transition':
            duration = row['info']['delta_frames']
            frame = row['info']['elapsed_frames'] - duration
            _, self.buffer = self.buffer.advance(row['action'], duration, self._delay(frame))
        return super().annotate(row)

    def _suffix(self, frame, held, remaining, action):
        if self.cached_frame is None:
            return ((action, remaining),) if held else ()
        skipped, phase = divmod(frame - self.cached_frame, HOLD)
        actions = self.cached_actions[skipped:]
        if held:
            # Continue the live hold. The suffix starts at its next decision
            # boundary. Ignore misaligned schedules after a boundary interrupt.
            tail = actions[1:] if actions and (phase + remaining == HOLD) else ()
            return ((action, remaining),) + tuple((a, HOLD) for a in tail)
        return tuple((a, HOLD) for a in actions) if phase == 0 else ()

    def _recheck(self, scene, schedule, frame, deadline):
        motion = []
        buffer = self.buffer
        for action, duration in schedule:
            moves, buffer = buffer.advance(action, duration, self._delay(frame + len(motion)))
            motion.extend(moves)
        model = SweptCollision(scene)
        start = np.asarray(scene.player)
        offset = 0
        for move, group in groupby(motion):
            if self.clock() >= deadline:
                return 'time_limit', None
            duration = len(list(group))
            end = start + VECTORS[move] * (scene.speed * duration)
            safe, _ = model.edges([start], [end], offset, duration)
            if not safe[0]:
                return 'collision', None
            start, offset = end, offset + duration
        horizon = self.config.get('cache_warning_lookahead_frames', 0)
        if len(schedule) > 1 and horizon and any(
                offset < laser.on < offset + horizon and laser.off > laser.on for laser in scene.lasers):
            if self.clock() >= deadline:
                return 'time_limit', None
            ttc = float(model.stationary_ttc([start], offset, horizon, warnings_only=True)[0])
            return ('warning_ttc' if ttc < horizon else 'clear'), ttc
        return 'clear', None

    def _baseline_action(self, obs, frame):
        if self.last is not None:
            raise RuntimeError('Missing transition annotation')
        age = frame - self.since if self.since is not None else None
        if age is not None and 0 <= age < HOLD:
            remaining = max(2, HOLD - age)
            player = np.asarray(obs['player'], dtype=np.float32)
            allowed = feasible_actions({'player': player}, remaining)
            if self.held_action in allowed:
                # The baseline only interrupts a live hold at the field boundary.
                self.decisions += 1
                self.last = {'committed_action': self.held_action, 'commitment_active': True,
                             'hold_age_frames': age, 'boundary_execution_allowed_actions': allowed,
                             'student_inference_seconds': 0., 'baseline_inference_skipped': True}
                return self.held_action
        return super().__call__(obs)

    def __call__(self, obs):
        started = self.clock()
        frame = self.elapsed_frames()
        action = self._baseline_action(obs, frame)
        baseline_finished = self.clock()
        deadline = baseline_finished + self.config['budget']['max_ms'] / 1000
        captured = self.capture_started() if self.capture_started else None
        if captured is not None:
            deadline = min(deadline, captured + self.config['capture_to_planner_deadline_ms'] / 1000)
        held = self.last['commitment_active']
        remaining = max(2, HOLD - self.last['hold_age_frames']) if held else HOLD
        schedule = self._suffix(frame, held, remaining, action)
        prefix = sum(duration for _, duration in schedule)
        can_reuse = bool(schedule) and (held or prefix >= self.config['minimum_fresh_reuse_frames'])
        record = {'baseline_action': action, 'suggested_action': None, 'source': 'skipped',
                  'stop_reason': 'time_limit', 'recheck': 'not_run', 'checked_edges': 0,
                  'route_actions': [], 'route_durations': [], 'checked_prefix_frames': 0,
                  'horizon_frames': None, 'leaf_ttc_frames': None, 'search_seconds': 0.,
                  'recheck_seconds': 0., 'baseline_seconds': baseline_finished - started,
                  'cache_reused': False}
        if 'cache_warning_lookahead_frames' in self.config:
            record['cache_warning_ttc_frames'] = None
        if self.clock() < deadline:
            scene, _ = scene_from_observation(obs)
            scene = replace(scene, player_radius=scene.player_radius + self.config['clearance_px'])
            if can_reuse:
                recheck_started = self.clock()
                status, warning_ttc = self._recheck(scene, schedule, frame, deadline)
                record.update(recheck=status, recheck_seconds=self.clock() - recheck_started)
                if 'cache_warning_ttc_frames' in record:
                    record['cache_warning_ttc_frames'] = warning_ttc
                if status in ('collision', 'warning_ttc'):
                    self.cached_actions = ()
                    self.cached_frame = self.cached_horizon = None
                if status == 'clear' or (held and status == 'warning_ttc'):
                    if status == 'warning_ttc':
                        # The executed hold was checked. Replan its suffix at the next boundary.
                        schedule, prefix = schedule[:1], remaining
                    reused = not held or len(schedule) > 1
                    record.update(source='cache' if reused else 'hold', cache_reused=reused,
                                  stop_reason='rechecked', horizon_frames=self.cached_horizon if reused else remaining,
                                  suggested_action=schedule[0][0], checked_prefix_frames=prefix,
                                  route_actions=[a for a, _ in schedule],
                                  route_durations=[duration for _, duration in schedule])
            refresh = record['recheck'] == 'collision' or (not held and record['recheck'] == 'warning_ttc')
            if (not can_reuse or refresh) and self.clock() < deadline:
                search_started = self.clock()
                reserve = self.config.get('leaf_reserve_ms', 0.) if any(
                    laser.off > laser.on > 0 for laser in scene.lasers) else 0.
                result = plan(scene, previous_action=self.buffer.requested, adaptive=True,
                              width=self.config['width'], budget=self.budget,
                              clock=self.clock, deadline=deadline, input_buffer=self.buffer,
                              switch_delay=self._delay(frame), leaf_reserve_ms=reserve)
                route = result.selected
                actions = route.actions if route else ()
                if actions:
                    self.cached_actions, self.cached_frame = actions, frame
                    self.cached_horizon = result.horizon
                record.update(source='search', stop_reason=result.stop_reason, checked_edges=result.checked_edges,
                              horizon_frames=result.horizon, suggested_action=actions[0] if actions else None,
                              checked_prefix_frames=len(actions) * HOLD, route_actions=list(actions),
                              route_durations=[HOLD] * len(actions),
                              leaf_ttc_frames=route.ttc if route and result.ttc_evaluated else None,
                              search_seconds=self.clock() - search_started)
        record['shadow_seconds'] = self.clock() - baseline_finished
        self.last['planner_shadow'] = record
        return action
