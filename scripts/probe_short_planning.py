"""Benchmark short-horizon planning on model rollouts and frozen trace states."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from agent.forward_model import Circle, ForwardModel, Scene, TimedLaser
from agent.planning_cases import micro_cases
from agent.replay_data import decision_record
from agent.short_planner import (
    EXIT_DELAY, EXIT_SLACK, HOLD, LONG_HORIZON, SHORT_HORIZON, SMOOTH_TOLERANCE,
    TTC_HORIZON, TTC_WEIGHT, TURN_WEIGHT, VECTORS, SearchBudget, plan, route_points,
)
from agent.swept_collision import SweptCollision
from agent.static_guidance import relative_goal, static_waypoints
from agent.timed_planner import TimedGoal
from scripts.probe_planning import trace_cases


MODES = {
    'short_clearance': {'ttc_weight': 0.},
    'short_ttc': {},
    'adaptive_ttc': {'adaptive': True},
}

BUDGET_MODES = {
    'spatial_ttc': {'cell_size': 8.},
    'wide_ttc': {'width': 48},
    'budgeted_short_ttc': {'budget': SearchBudget(378, 5.)},
    'budgeted_adaptive_ttc': {'adaptive': True, 'budget': SearchBudget(900, 8.)},
}


def benchmark_cases():
    return micro_cases() + [
        {'name': 'far_fast_circle',
         'scene': Scene((8., 20.), circles=(Circle(100., 20., -20., 0., 2.),),
                        bounds=(0., 40., 0., 40.)),
         'goal': TimedGoal(8., 36., 0, 24, 1.), 'horizon': 24, 'expect_arrival': True},
        {'name': 'required_reversal',
         'scene': Scene((16., .5), lasers=(TimedLaser(4, 0, 4, 1, 22, 4, 7.5),
                                          TimedLaser(28, 0, 28, 1, 22, 12, 32)),
                        bounds=(0., 32., 0., 1.)), 'horizon': 24},
        {'name': 'evacuate_before_short_horizon_sees_laser',
         'scene': Scene((40., 40.), lasers=(TimedLaser(40, 0, 40, 80, 58, 18, 40),),
                        bounds=(0., 80., 0., 80.)), 'horizon': 32},
    ]


def _timings(seconds):
    ms = np.array(seconds) * 1000
    return {'samples': len(ms), 'median_ms': float(np.median(ms)),
            'p95_ms': float(np.percentile(ms, 95)), 'max_ms': float(ms.max())}


def _verify(scene, route):
    """Recheck the whole returned route against the unfiltered reference."""
    points = route_points(scene, route)
    valid = np.allclose(points[-1], route.point, rtol=0, atol=1e-9)
    reference = ForwardModel(scene)
    duration = 1 if route.motion else HOLD
    for index, (start, end) in enumerate(zip(points, points[1:])):
        allowed, _ = reference.edges([start], [end], index * duration, duration)
        valid &= bool(allowed[0])
    if not valid:
        raise AssertionError('Returned route failed reference replay')
    return points


def _advance(scene, player):
    circles = tuple(replace(c, x=c.x + c.vx * HOLD, y=c.y + c.vy * HOLD) for c in scene.circles)
    lasers = tuple(replace(l, on=max(0., l.on - HOLD), off=l.off - HOLD)
                   for l in scene.lasers if l.off > HOLD)
    return replace(scene, player=tuple(map(float, player)), circles=circles, lasers=lasers)


def _search_details(result):
    return {'stop_reason': result.stop_reason, 'checked_edges': result.checked_edges,
            'proposed_first_action': result.proposed.actions[0] if result.proposed else None,
            'selected_first_action': result.selected.actions[0] if result.selected else None,
            'ttc_evaluated': result.ttc_evaluated, 'layers': [asdict(layer) for layer in result.layers],
            'candidates': [asdict(route) for route in result.candidates]}


def _compare_coverage(search, reference_actions):
    layer = next((layer for layer in search['layers'] if layer['frame'] == SHORT_HORIZON), None)
    retained = set(layer['retained']) if layer else set()
    reference = set(reference_actions)
    if retained - reference:
        raise AssertionError('Beam produced a first action outside the unpruned reference')
    return {'evaluated_16_layer': layer is not None, 'reference_first_actions': sorted(reference),
            'retained_first_actions': sorted(retained), 'missing_first_actions': sorted(reference - retained)}


def _reference_actions(case):
    previous = case.get('metadata', {}).get('prior_action') or 0
    result = plan(case['scene'], case.get('goal'), previous_action=previous, width=None, ttc_weight=0.)
    return result.first_actions_at(SHORT_HORIZON)


def _rollout(case, settings, *, guide_static=False):
    scene, goal = case['scene'], case.get('goal')
    started = time.perf_counter()
    waypoints = static_waypoints(scene, goal) if guide_static else ()
    guide_ms = (time.perf_counter() - started) * 1000 if guide_static else None
    waypoint = 0
    arrived = 0 if goal is not None and goal.reached(scene.player, 0) else None
    frame, previous = 0, 0
    seconds, decisions = [], []
    stop_reason = 'completed'
    while frame < case['horizon']:
        while waypoint < len(waypoints) and waypoints[waypoint].reached(scene.player, frame):
            waypoint += 1
        target = waypoints[waypoint] if waypoint < len(waypoints) else goal
        target = relative_goal(target, frame) if arrived is None else None
        start = time.perf_counter()
        result = plan(scene, target, previous_action=previous, **settings)
        seconds.append(time.perf_counter() - start)
        route = result.selected
        decisions.append({'frame': frame, 'horizon': result.horizon,
                          'hint': asdict(result.hint) if result.hint else None,
                          'route': list(route.actions) if route else None, 'smoothed': result.smoothed,
                          'leaf_ttc': route.ttc if route and result.ttc_evaluated else None,
                          'search': _search_details(result),
                          'replay': decision_record(scene, frame, target, result=result,
                                                    action=route.actions[0] if route else None)})
        if route is None:
            stop_reason = 'no_valid_action'
            break
        points = _verify(scene, route)
        scene = _advance(scene, points[1])
        previous = route.actions[0]
        frame += HOLD
        if goal is not None and arrived is None and goal.reached(scene.player, frame):
            arrived = frame
    objective_met = goal is None or (arrived is not None) == case['expect_arrival']
    return {'pass': frame == case['horizon'] and objective_met, 'completed_frames': frame,
            'goal_arrival_frame': arrived, 'objective_met': objective_met, 'stop_reason': stop_reason,
            'decisions': decisions, 'timing': _timings(seconds), 'guide_ms': guide_ms,
            'waypoints': [asdict(point) for point in waypoints]}


def _audit_collision(scene):
    """Compare the two kernels on identical legal/illegal edges across time layers."""
    offsets = np.array([(x, y) for x in (-16, -8, 0, 8, 16) for y in (-16, -8, 0, 8, 16)])
    starts = np.repeat(np.asarray(scene.player) + offsets, len(VECTORS), axis=0)
    ends = starts + np.tile(VECTORS * scene.speed * HOLD, (len(offsets), 1))
    reference, swept = ForwardModel(scene), SweptCollision(scene)
    reference_seconds = swept_seconds = 0.
    largest_error = 0.
    for frame in range(0, LONG_HORIZON, HOLD):
        start = time.perf_counter()
        expected, reference_gaps = reference.edges(starts, ends, frame, HOLD)
        reference_seconds += time.perf_counter() - start
        start = time.perf_counter()
        actual, swept_gaps = swept.edges(starts, ends, frame, HOLD)
        swept_seconds += time.perf_counter() - start
        if not np.array_equal(expected, actual):
            raise AssertionError('Swept AABB validity differs from reference')
        if expected.any():
            error = np.max(np.abs(np.minimum(reference_gaps[expected], swept.clearance_cap) - swept_gaps[expected]))
            largest_error = max(largest_error, float(error))
    if largest_error > 1e-8:
        raise AssertionError('Swept AABB changed capped clearance')
    return {'edges': len(starts) * (LONG_HORIZON // HOLD), 'validity_mismatches': 0,
            'max_valid_capped_gap_error': largest_error, 'reference_ms': reference_seconds * 1000,
            'swept_ms': swept_seconds * 1000, **asdict(swept.counts)}


def _snapshot(case, settings, repeats):
    scene = case['scene']
    previous = case['metadata'].get('prior_action') or 0
    seconds, attempts = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        result = plan(scene, previous_action=previous, **settings)
        seconds.append(time.perf_counter() - start)
        route = result.selected
        frames = 0 if route is None else len(route.actions) * HOLD
        if route is not None:
            _verify(scene, route)
        attempts.append({'elapsed_ms': seconds[-1] * 1000, 'stop_reason': result.stop_reason,
                         'safe_prefix_frames': frames, 'checked_edges': result.checked_edges,
                         'first_actions_16': result.first_actions_at(SHORT_HORIZON)})
    return {'pass': all(a['safe_prefix_frames'] >= SHORT_HORIZON for a in attempts),
            'complete_requested_horizon': all(a['safe_prefix_frames'] == result.horizon for a in attempts),
            'safe_prefix_frames': frames, 'requested_horizon': result.horizon,
            'exit_hint': asdict(result.hint) if result.hint else None,
            'previous_action': previous, 'smoothed': result.smoothed,
            'actions': list(route.actions) if route else [],
            'leaf_ttc': route.ttc if route and result.ttc_evaluated else None,
            'clearance': route.clearance if route else None,
            'turn_degrees': float(np.degrees(route.turn)) if route else None,
            'collision_counts': asdict(result.counts), 'timing': _timings(seconds),
            'search': _search_details(result), 'attempts': attempts,
            'replay': decision_record(scene, 0, result=result,
                                      action=route.actions[0] if route else None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--state', action='append', default=[], help='Fresh EPISODE:STEP')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--audit-budget', action='store_true', help='Compare bounded search and first-action coverage')
    parser.add_argument('--guide-static', action='store_true', help='Compare verified static subgoals')
    parser.add_argument('--output', type=Path, default=Path('runs/short-planning'))
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20 or bool(args.trace) != bool(args.state):
        parser.error('Use --trace with --state and 1..20 repeats')
    try:
        states = [tuple(map(int, state.split(':'))) for state in args.state]
        if any(len(state) != 2 or min(state) < 0 for state in states) or len(set(states)) != len(states):
            raise ValueError()
    except ValueError:
        parser.error('States must be unique nonnegative EPISODE:STEP pairs')
    trace, digest = trace_cases(args.trace, states, LONG_HORIZON) if states else ([], None)
    modes = {**MODES, **BUDGET_MODES} if args.audit_budget else dict(MODES)
    if args.guide_static:
        modes['static_guided_ttc'] = {}
    output = args.output
    suffix = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            output = args.output.with_name(f'{args.output.name}-{suffix:03d}')
    report = {
        'planner_contract': 'short-nine-swept-v2',
        'configuration': {'hold': HOLD, 'short_horizon': SHORT_HORIZON, 'long_horizon': LONG_HORIZON,
                          'ttc_horizon': TTC_HORIZON, 'ttc_weight': TTC_WEIGHT, 'turn_weight': TURN_WEIGHT,
                          'exit_slack': EXIT_SLACK, 'exit_delay': EXIT_DELAY,
                          'smooth_score_tolerance': SMOOTH_TOLERANCE, 'width': 16, 'repeats': args.repeats},
        'mode_settings': {name: {key: asdict(value) if isinstance(value, SearchBudget) else value
                                 for key, value in settings.items()} for name, settings in modes.items()},
        'trace': {'path': str(args.trace.resolve()), 'sha256': digest} if trace else None,
        'pass_definition': {'micro': 'Complete model rollout and satisfy the declared arrival expectation',
                            'trace': 'Find a reference-checked prefix of at least 16 frames in every repetition'},
        'timing_scope': 'Planner call including trigger and TTC. Excludes decoding, reference replay and game IO.',
        'model_limits': 'Known constant-velocity circles and timed static lasers. Frozen trace counterfactuals.',
        'micro': [], 'snapshots': [],
    }
    for case in benchmark_cases():
        results = {name: _rollout(case, settings, guide_static=name == 'static_guided_ttc')
                   for name, settings in modes.items()}
        if args.audit_budget:
            reference = _reference_actions(case)
            for result in results.values():
                result['initial_coverage'] = _compare_coverage(result['decisions'][0]['search'], reference)
        report['micro'].append({'name': case['name'], 'scene': asdict(case['scene']),
                                'goal': asdict(case['goal']) if case.get('goal') else None,
                                'expect_arrival': case.get('expect_arrival'), 'horizon': case['horizon'],
                                'collision_audit': _audit_collision(case['scene']),
                                'modes': results})
    for case in trace:
        results = {name: _snapshot(case, settings, args.repeats) for name, settings in modes.items()}
        if args.audit_budget:
            reference = _reference_actions(case)
            for result in results.values():
                result['coverage'] = _compare_coverage(result['search'], reference)
        report['snapshots'].append({'name': case['name'], 'metadata': case['metadata'],
                                    'scene': asdict(case['scene']),
                                    'collision_audit': _audit_collision(case['scene']),
                                    'modes': results})
    if args.guide_static:
        scene = Scene((20., 20.), bounds=(0., 40., 0., 40.))
        ticks = iter((0., 0., .002, .002))
        result = plan(scene, budget=SearchBudget(378, 1.), clock=lambda: next(ticks))
        report['diagnostic_tracks'] = [{
            'label': 'Time cutoff / checked prefix', 'scene': asdict(scene),
            'source': 'Injected clock diagnostic. Not a measured runtime sample.',
            'frames': [decision_record(scene, 0, result=result, action=result.selected.actions[0])],
        }]
    report['pass_counts'] = {group: {mode: sum(case['modes'][mode]['pass'] for case in report[group])
                                      for mode in modes} for group in ('micro', 'snapshots')}
    target = output / 'report.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = [f'Report: {target.resolve()}']
    for group in ('micro', 'snapshots'):
        if not report[group]:
            continue
        for mode, count in report['pass_counts'][group].items():
            lines.append(f'{group} {mode}: {count}/{len(report[group])} passed')
    (output / 'planning.console.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return output


if __name__ == '__main__':
    main()
