"""Benchmark short-horizon planning on model rollouts and frozen trace states."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from agent.forward_model import Circle, ForwardModel, Scene, TimedLaser
from agent.planning_cases import micro_cases
from agent.short_planner import (
    EXIT_DELAY, EXIT_SLACK, HOLD, LONG_HORIZON, SHORT_HORIZON, SMOOTH_TOLERANCE,
    TTC_HORIZON, TTC_WEIGHT, TURN_WEIGHT, VECTORS, plan, route_points,
)
from agent.swept_collision import SweptCollision
from agent.timed_planner import TimedGoal
from scripts.probe_planning import trace_cases


MODES = {
    'short_clearance': {'ttc_weight': 0.},
    'short_ttc': {},
    'adaptive_ttc': {'adaptive': True},
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
    for index, (start, end) in enumerate(zip(points, points[1:])):
        allowed, _ = reference.edges([start], [end], index * HOLD, HOLD)
        valid &= bool(allowed[0])
    if not valid:
        raise AssertionError('Returned route failed reference replay')
    return points


def _advance(scene, player):
    circles = tuple(replace(c, x=c.x + c.vx * HOLD, y=c.y + c.vy * HOLD) for c in scene.circles)
    lasers = tuple(replace(l, on=max(0., l.on - HOLD), off=l.off - HOLD)
                   for l in scene.lasers if l.off > HOLD)
    return replace(scene, player=tuple(map(float, player)), circles=circles, lasers=lasers)


def _rollout(case, settings):
    scene, goal = case['scene'], case.get('goal')
    arrived = 0 if goal is not None and goal.reached(scene.player, 0) else None
    frame, previous = 0, 0
    seconds, decisions = [], []
    stop_reason = 'completed'
    while frame < case['horizon']:
        relative_goal = None
        if goal is not None and arrived is None and frame <= goal.deadline:
            relative_goal = replace(goal, earliest=max(0, goal.earliest - frame), deadline=goal.deadline - frame)
        start = time.perf_counter()
        result = plan(scene, relative_goal, previous_action=previous, **settings)
        seconds.append(time.perf_counter() - start)
        route = result.selected
        decisions.append({'frame': frame, 'horizon': result.horizon,
                          'hint': asdict(result.hint) if result.hint else None,
                          'route': list(route.actions) if route else None, 'smoothed': result.smoothed,
                          'leaf_ttc': route.ttc if route and settings.get('ttc_weight', TTC_WEIGHT) else None})
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
            'decisions': decisions, 'timing': _timings(seconds)}


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
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = plan(scene, previous_action=previous, **settings)
        seconds.append(time.perf_counter() - start)
    route = result.selected
    frames = 0 if route is None else len(route.actions) * HOLD
    if route is not None:
        _verify(scene, route)
    return {'pass': frames >= SHORT_HORIZON, 'complete_requested_horizon': frames == result.horizon,
            'safe_prefix_frames': frames, 'requested_horizon': result.horizon,
            'exit_hint': asdict(result.hint) if result.hint else None,
            'previous_action': previous, 'smoothed': result.smoothed,
            'actions': list(route.actions) if route else [],
            'leaf_ttc': route.ttc if route and settings.get('ttc_weight', TTC_WEIGHT) else None,
            'clearance': route.clearance if route else None,
            'turn_degrees': float(np.degrees(route.turn)) if route else None,
            'collision_counts': asdict(result.counts), 'timing': _timings(seconds)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--state', action='append', default=[], help='Fresh EPISODE:STEP')
    parser.add_argument('--repeats', type=int, default=7)
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
        'planner_contract': 'short-nine-swept-v1',
        'configuration': {'hold': HOLD, 'short_horizon': SHORT_HORIZON, 'long_horizon': LONG_HORIZON,
                          'ttc_horizon': TTC_HORIZON, 'ttc_weight': TTC_WEIGHT, 'turn_weight': TURN_WEIGHT,
                          'exit_slack': EXIT_SLACK, 'exit_delay': EXIT_DELAY,
                          'smooth_score_tolerance': SMOOTH_TOLERANCE, 'width': 16, 'repeats': args.repeats},
        'trace': {'path': str(args.trace.resolve()), 'sha256': digest} if trace else None,
        'pass_definition': {'micro': 'Complete model rollout and satisfy the declared arrival expectation',
                            'trace': 'Find a reference-checked prefix of at least 16 frames'},
        'timing_scope': 'Planner call including trigger and TTC. Excludes decoding, reference replay and game IO.',
        'model_limits': 'Known constant-velocity circles and timed static lasers. Frozen trace counterfactuals.',
        'micro': [], 'snapshots': [],
    }
    for case in benchmark_cases():
        report['micro'].append({'name': case['name'], 'scene': asdict(case['scene']),
                                'goal': asdict(case['goal']) if case.get('goal') else None,
                                'expect_arrival': case.get('expect_arrival'), 'horizon': case['horizon'],
                                'collision_audit': _audit_collision(case['scene']),
                                'modes': {name: _rollout(case, settings) for name, settings in MODES.items()}})
    for case in trace:
        report['snapshots'].append({'name': case['name'], 'metadata': case['metadata'],
                                    'collision_audit': _audit_collision(case['scene']),
                                    'modes': {name: _snapshot(case, settings, args.repeats)
                                              for name, settings in MODES.items()}})
    report['pass_counts'] = {group: {mode: sum(case['modes'][mode]['pass'] for case in report[group])
                                      for mode in MODES} for group in ('micro', 'snapshots')}
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
