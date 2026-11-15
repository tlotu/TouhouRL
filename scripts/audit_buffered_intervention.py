"""Audit fresh buffered shadow proposals against recorded laser observations."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from functools import cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agent.boundary_control import feasible_actions
from agent.direction_buffer import DirectionBuffer
from agent.forward_model import ForwardModel, scene_from_observation
from agent.laser_shadow import BUFFERED_LEAF_CONFIG, BUFFERED_REUSE_CONFIG
from agent.short_planner import HOLD, SHORT_HORIZON, TTC_HORIZON, VECTORS
from agent.swept_collision import SweptCollision
from env.action_emulator import DIRECTIONS
from env.windows import INPUT_TIMING_CONTRACT
from scripts.analyze_laser_shadow import timing_ms


DEADLINE = .0166
PADDING = BUFFERED_LEAF_CONFIG['clearance_px']


def transitions(path, *, allow_earlier=False):
    with path.open('rb') as stream:
        header = json.loads(next(stream))
        configs = (BUFFERED_LEAF_CONFIG, BUFFERED_REUSE_CONFIG) if allow_earlier else (BUFFERED_LEAF_CONFIG,)
        if (header.get('planner_shadow') not in configs
                or header.get('input_timing_contract') not in (None, INPUT_TIMING_CONTRACT)):
            raise ValueError('Expected a sequential buffered leaf shadow log')
        rows = None
        for raw in stream:
            row = json.loads(raw)
            if row['event'] == 'reset':
                if rows is not None:
                    raise ValueError('Reset before completing the previous episode')
                rows = []
            elif row['event'] == 'transition':
                if rows is None:
                    raise ValueError('Transition outside an episode')
                rows.append(row)
            elif row['event'] == 'episode_end':
                if not rows:
                    raise ValueError('Episode ended without transitions')
                yield header, rows, row['reason']
                rows = None
            elif row['event'] == 'error':
                raise ValueError('Source log contains an error')
        if rows is not None:
            raise ValueError('Source log ended before completing an episode')


def dispatch_budgets(path):
    samples = defaultdict(list)
    for header, rows, _ in transitions(path):
        if header.get('input_timing_contract') != INPUT_TIMING_CONTRACT:
            raise ValueError('Expected a sequential input profile')
        for row in rows:
            timing = row['info']['timing']
            if timing['previous_action'] is not None:
                samples[len(timing['input_dispatch']['events'])].append(timing['input_send_seconds'])
    return {n: timing_ms(times) for n, times in sorted(samples.items())}


def motions(buffers, actions):
    """Enumerate distinct motion and queue states under zero or one frame of start delay."""
    paths = {((), buffer) for buffer in buffers}
    for action in actions:
        paths = {(path + moves, tail) for path, buffer in paths for delay in (0, 1)
                 for moves, tail in (buffer.advance(action, HOLD, delay),)}
    return paths


def path_points(model, paths, *, clip=False):
    moves = np.asarray([path for path, _ in paths])
    steps = model.scene.speed * VECTORS[moves]
    if not clip:
        return model.scene.player + steps.cumsum(axis=1)
    xmin, xmax, ymin, ymax = model.scene.bounds
    point = np.broadcast_to(model.scene.player, (len(paths), 2))
    points = []
    for step in steps.transpose(1, 0, 2):
        point = np.clip(point + step, (xmin, ymin), (xmax, ymax))
        points.append(point)
    return np.stack(points, axis=1)


def safe_paths(model, points):
    valid = np.ones(len(points), dtype=bool)
    starts = np.broadcast_to(model.scene.player, (len(points), 2))
    for frame, ends in enumerate(points.transpose(1, 0, 2)):
        safe, _ = model.edges(starts, ends, frame, 1)
        valid &= safe
        starts = ends
    return valid


def warning_exposure(scene, buffers, actions):
    paths = list(motions(buffers, (*actions, 0)))
    if any(buffer.pending or buffer.applied for _, buffer in paths):
        return None
    model = SweptCollision(scene)
    points = path_points(model, paths, clip=True)
    frames = points.shape[1]
    remaining = TTC_HORIZON - frames
    times = model.stationary_ttc(points[:, -1], frames, remaining)
    return ~safe_paths(model, points) | (times < remaining)


def screen(scene, buffers, baseline, actions):
    padded = replace(scene, player_radius=scene.player_radius + PADDING)
    model = SweptCollision(padded)
    alternatives = path_points(model, list(motions(buffers, actions)))
    if not safe_paths(model, alternatives).all():
        return 'alternative_not_robust', alternatives
    current = path_points(model, list(motions(buffers, (baseline,))), clip=True)
    if not safe_paths(model, current).any():
        return 'baseline_first_hold_collision', alternatives
    warnings = tuple(laser for laser in padded.lasers if laser.off > laser.on > 0)
    if warnings:
        warning_scene = replace(padded, circles=(), lasers=warnings)
        exposure = warning_exposure(warning_scene, buffers, (baseline,))
        if exposure is not None and exposure.all():
            escape = warning_exposure(warning_scene, buffers, actions)
            if escape is not None and not escape.any():
                return 'warning_escape', alternatives
    return 'no_clear_benefit', alternatives


def frame_start(row):
    return row['info']['elapsed_frames'] - row['info']['delta_frames']


def rollout(rows, models, start, buffer, delay, actions=()):
    """Follow a fixed branch through refreshed, recorded hazards."""
    origin = frame_start(rows[start])
    position = np.asarray(models(start).scene.player)
    contacts = [None, None]
    elapsed = 0
    omitted = 0
    for index in range(start, len(rows)):
        row = rows[index]
        scene = models(index).scene
        bounds = scene.bounds
        lower, upper = (bounds[0], bounds[2]), (bounds[1], bounds[3])
        omitted = max(omitted, rows[index - 1]['info']['dropped_bullets'])
        for offset in range(min(row['info']['delta_frames'], TTC_HORIZON - elapsed)):
            hold = elapsed // HOLD
            requested = actions[hold] if hold < len(actions) else row['action']
            (move,), buffer = buffer.advance(requested, 1, delay)
            end = np.clip(position + scene.speed * VECTORS[move], lower, upper)
            _, gaps = models(index).edges([position], [end], offset, 1)
            for i, margin in enumerate((0., PADDING)):
                if contacts[i] is None and gaps[0] <= margin:
                    contacts[i] = elapsed + 1
            position = end
            elapsed += 1
        if elapsed == TTC_HORIZON:
            break
    return {'through_frame': origin + elapsed, 'first_contact_offset_frames': contacts,
            'max_dropped_bullets': omitted}


def compare_contacts(replay, frame, baseline='recorded_requests'):
    """Compare first model contacts, keeping ties at the end of the window."""
    comparisons = {}
    for branch in replay:
        if branch == baseline:
            continue
        by_margin = {}
        for i, margin in enumerate(('nominal', 'padded')):
            differences = []
            for old, new in zip(replay[baseline], replay[branch]):
                end = old['through_frame'] - frame + 1
                differences.append((new['first_contact_offset_frames'][i] or end)
                                   - (old['first_contact_offset_frames'][i] or end))
            if min(differences) > 0:
                label = 'later_both'
            elif max(differences) < 0:
                label = 'earlier_both'
            elif min(differences) < 0 < max(differences):
                label = 'mixed'
            elif min(differences) < 0:
                label = 'earlier_one'
            elif max(differences) > 0:
                label = 'later_one'
            else:
                label = 'unchanged'
            by_margin[margin] = {'classification': label, 'difference_frames_by_delay': differences}
        comparisons[branch] = by_margin
    return comparisons


def audit(path, budgets):
    counts = Counter()
    candidates = []
    header = None
    for header, rows, outcome in transitions(path):
        @cache
        def models(index):
            return ForwardModel(scene_from_observation(rows[index]['observation'])[0])

        buffers = {DirectionBuffer()}
        constant = [DirectionBuffer(), DirectionBuffer()]
        prior = None
        for index, row in enumerate(rows):
            info, shadow = row['info'], row['planner_shadow']
            timing = info['timing']
            frame, baseline = frame_start(row), row['action']
            if (timing['applied_action'] != baseline or baseline not in row['boundary_execution_allowed_actions']
                    or timing['previous_action'] != (prior['action'] if prior else None)):
                raise ValueError('Recorded execution is inconsistent')
            if prior is not None and (prior['next_observation'] != row['observation']
                                      or prior['info']['elapsed_frames'] != frame):
                raise ValueError('Unaligned replay observations')
            proposal = shadow['suggested_action']
            counts['observed_transitions'] += 1
            if proposal is not None and proposal != baseline:
                counts['changed_proposal'] += 1
                if row['commitment_active']:
                    counts['held'] += 1
                elif proposal not in row['boundary_execution_allowed_actions'] or proposal not in feasible_actions(
                        row['observation'], HOLD):
                    counts['execution_boundary'] += 1
                elif shadow['checked_prefix_frames'] < SHORT_HORIZON:
                    counts['short_prefix'] += 1
                elif prior is None:
                    counts['missing_capture_time'] += 1
                else:
                    actions = tuple(shadow['route_actions'][:SHORT_HORIZON // HOLD])
                    if (len(actions) * HOLD != SHORT_HORIZON or actions[0] != proposal
                            or shadow['route_durations'][:len(actions)] != [HOLD] * len(actions)):
                        raise ValueError('Invalid fresh route')
                    before = prior['info']['timing']['sample_read_seconds'] + timing['observation_age_seconds']
                    previous = timing['previous_action']
                    events = len(set(DIRECTIONS[previous]) ^ set(DIRECTIONS[proposal]))
                    send = budgets[events]['p95'] / 1000
                    counts['structural_pass'] += 1
                    scene = models(index).scene
                    started = time.perf_counter()
                    reason, alternatives = screen(scene, buffers, baseline, actions)
                    guard = time.perf_counter() - started
                    counts[reason] += 1
                    if reason in ('warning_escape', 'baseline_first_hold_collision'):
                        if not safe_paths(ForwardModel(replace(scene, player_radius=scene.player_radius + PADDING)),
                                          alternatives).all():
                            raise ValueError('Swept and reference checks disagree')
                        without_guard = before + send
                        with_guard = without_guard + guard
                        counts['benefit_within_dispatch_budget'] += without_guard <= DEADLINE
                        counts['benefit_within_budget_with_guard'] += with_guard <= DEADLINE
                        branches = {'recorded_requests': (), 'one_hold_override': actions[:1],
                                    'certified_prefix': actions}
                        replay = {name: [rollout(rows, models, index, constant[delay], delay, schedule)
                                         for delay in (0, 1)] for name, schedule in branches.items()}
                        candidates.append({'episode': row['episode'] + 1, 'episode_outcome': outcome, 'frame': frame,
                                           'baseline': baseline, 'proposal': proposal, 'reason': reason,
                                           'prefix_frames': shadow['checked_prefix_frames'],
                                           'alternative_key_events': events, 'capture_to_dispatch_ms': 1000 * before,
                                           'estimated_dispatch_ms': 1000 * send, 'guard_ms': 1000 * guard,
                                           'projected_ms': 1000 * without_guard,
                                           'projected_with_guard_ms': 1000 * with_guard, 'replay': replay,
                                           'contact_comparison': compare_contacts(replay, frame)})
            for delay in (0, 1):
                _, constant[delay] = constant[delay].advance(baseline, info['delta_frames'], delay)
            buffers = {buffer.advance(baseline, info['delta_frames'], delay)[1]
                       for buffer in buffers for delay in (0, 1)}
            prior = row
        models.cache_clear()
    if header is None:
        raise ValueError('No complete episodes')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'source': str(path), 'sha256': digest, 'model_sha256': header['model_sha256'],
            'counts': dict(counts), 'candidates': candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--timing-profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    budgets = dispatch_budgets(args.timing_profile)
    report = {'dispatch_ms_by_key_events': budgets, 'timing_profile': str(args.timing_profile),
              'hold_frames': HOLD, 'minimum_prefix_frames': SHORT_HORIZON, 'padding_px': PADDING,
              'counterfactual_horizon_frames': TTC_HORIZON,
              'limitations': [
                  'Dispatch times are empirical p95 by number of changed keys, including sparse groups.',
                  'Guard time is offline geometry work; the projection retains historical acquisition and planner time.',
                  'Branches reuse observed hazard snapshots and revert to recorded requests after the override.',
                  'Counterfactual movement and collision model contacts are not live outcomes.',
              ],
              'traces': [audit(path, budgets) for path in args.traces]}
    candidates = [candidate for trace in report['traces'] for candidate in trace['candidates']]
    report['contact_summary'] = {}
    for name, subset in (
            ('all', candidates),
            ('within_budget_with_guard', [c for c in candidates if c['projected_with_guard_ms'] <= 1000 * DEADLINE])):
        report['contact_summary'][name] = {
            branch: {margin: dict(Counter(c['contact_comparison'][branch][margin]['classification'] for c in subset))
                     for margin in ('nominal', 'padded')}
            for branch in ('one_hold_override', 'certified_prefix')}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for trace in report['traces']:
        print(trace['source'], trace['counts'])


if __name__ == '__main__':
    main()
