"""Audit laser controls, timings and the first observed laser partition."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from agent.laser_contract import LASER_SCHEMA, LASER_TASK
from agent.laser_cached import CACHE_CONFIG, intervention_reason
from agent.input_switch import SWITCH_GUARD
from agent.laser_policy import decode_lasers
from agent.laser_room import partition_room
from agent.laser_shadow import BUFFERED_LEAF_CONFIG, BUFFERED_REUSE_CONFIG, BUFFERED_TIMING_CONFIG, SHADOW_CONFIG
from agent.short_planner import HOLD, LONG_HORIZON, SHORT_HORIZON
from env.action_emulator import DIRECTIONS
from env.windows import INPUT_TIMING_CONTRACT, MULTI_KEY_INPUT_TIMING_CONTRACT


def timing_ms(seconds):
    if not seconds:
        return {'count': 0}
    p50, p95, maximum = map(float, np.percentile(seconds, [50, 95, 100]) * 1000)
    return {'count': len(seconds), 'p50': p50, 'p95': p95, 'max': maximum,
            'at_least_nominal_frame': sum(value >= 1 / 60 for value in seconds)}


def first_partition(rows):
    """Find the first warning volley observed with all visible beams firing."""
    first_warning, peak = None, 0
    for frame, obs in rows:
        phases = Counter(int(laser[5]) for laser in obs['lasers'] if laser[11])
        warning, active = phases[0], phases[1]
        if first_warning is None:
            if warning:
                first_warning = frame
            else:
                continue
        peak = max(peak, active + warning)
        if warning:
            continue
        if active != peak or phases[2]:
            break
        px, py, *_ = obs['player']
        player = [(float(px) + 1) * 192, (float(py) + 1) * 224]
        room = partition_room(player, decode_lasers(obs['lasers']))
        return {'first_warning_frame': first_warning, 'full_activation_frame': frame,
                'active_lasers': active, 'player': player, 'seed_free': room['seed_free'],
                'area_pixels2': room['area_pixels2'] if room['seed_free'] else None}
    return {'first_warning_frame': first_warning, 'full_activation_frame': None,
            'max_visible_warning_and_active': peak}


def _dispatch_sample(row, pipeline, contract=INPUT_TIMING_CONTRACT):
    info = row['info']
    timing = info['timing']
    dispatch = timing['input_dispatch']
    old = timing['previous_action']
    before = set() if old is None else {'shift', 'z', *DIRECTIONS[old]}
    wanted = {'shift', 'z', *DIRECTIONS[row['action']]}
    releases, presses = sorted(before - wanted), sorted(wanted - before)
    expected = [(key, False) for key in releases] + [(key, True) for key in presses]
    events, checks = dispatch['events'], dispatch['foreground_seconds']
    batch = contract == MULTI_KEY_INPUT_TIMING_CONTRACT and old is not None and len(expected) > 1
    if ([(event['key'], event['down']) for event in events] != expected
            or ('batch' in dispatch) != batch
            or len(checks) != (2 if batch else 1 + 2 * len(presses))):
        raise ValueError('Unexpected input dispatch')
    calls = [dispatch['batch']] if batch else events
    total = timing['input_send_seconds']
    parts = [*checks, *(call[key] for call in calls for key in ('prepare_seconds', 'send_input_seconds'))]
    if any(not math.isfinite(value) or value < 0 for value in [total, *parts]) or sum(parts) > total + 1e-9:
        raise ValueError('Invalid input dispatch timings')
    return {'episode': row['episode'], 'frame': info['elapsed_frames'] - info['delta_frames'],
            'fresh': not row['commitment_active'], 'cache': row['planner_shadow']['source'] == 'cache',
            'capture_to_send_seconds': pipeline, 'dispatch_seconds': total,
            'foreground_seconds': sum(checks), 'foreground_checks': checks, 'events': events,
            'batched': batch, 'direction_switch': old is not None and old != row['action'],
            'prepare_seconds': sum(call['prepare_seconds'] for call in calls),
            'send_input_seconds': sum(call['send_input_seconds'] for call in calls),
            'native_call_seconds': [call['send_input_seconds'] for call in calls],
            'other_seconds': max(0., total - sum(parts))}


def _input_timing_report(samples, contract=INPUT_TIMING_CONTRACT):
    by_count = defaultdict(list)
    for sample in samples:
        by_count[len(sample['events'])].append(sample['dispatch_seconds'])
    fresh = [sample['capture_to_send_seconds'] for sample in samples
             if sample['fresh'] and sample['capture_to_send_seconds'] is not None]
    cache = [sample['capture_to_send_seconds'] for sample in samples
             if sample['cache'] and sample['capture_to_send_seconds'] is not None]
    switches = [sample for sample in samples if sample['direction_switch'] and len(sample['events']) > 1]
    switch_pipeline = [sample['capture_to_send_seconds'] for sample in switches
                       if sample['capture_to_send_seconds'] is not None]
    return {'contract': contract,
            'phase_ms': {key.removesuffix('_seconds'): timing_ms([sample[key] for sample in samples])
                         for key in ('dispatch_seconds', 'foreground_seconds', 'prepare_seconds',
                                     'send_input_seconds', 'other_seconds')},
            'dispatch_ms_by_key_events': {str(count): timing_ms(values) for count, values in sorted(by_count.items())},
            'foreground_check_ms': timing_ms([value for sample in samples for value in sample['foreground_checks']]),
            'send_input_call_ms': timing_ms([value for sample in samples for value in sample['native_call_seconds']]),
            'batched_dispatches': sum(sample['batched'] for sample in samples),
            'multi_key_switch_dispatch_ms': timing_ms([sample['dispatch_seconds'] for sample in switches]),
            'multi_key_switch_capture_to_send_ms': timing_ms(switch_pipeline),
            'multi_key_switch_capture_to_send_over_16_6ms': sum(value > .0166 for value in switch_pipeline),
            'fresh_capture_to_send_ms': timing_ms(fresh),
            'fresh_capture_to_send_over_16_6ms': sum(value > .0166 for value in fresh),
            'cache_capture_to_send_ms': timing_ms(cache),
            'cache_capture_to_send_over_10ms': sum(value > .010 for value in cache),
            'dispatch_tail_above_2ms': sorted((sample for sample in samples if sample['dispatch_seconds'] > .002),
                                               key=lambda sample: sample['dispatch_seconds'], reverse=True)}


def summarize_episode(rows, end, shadow_config, cached_mode=None, switch_guard=None, input_samples=None,
                      input_contract=None):
    times = defaultdict(list)
    counts, stops, deltas = Counter(), Counter(), Counter()
    partition_samples = []
    previous_info = None
    for row in rows:
        info = row['info']
        timing = info.get('timing', {})
        delta = info['delta_frames']
        partition_samples.append((info['elapsed_frames'] - delta, row['observation']))
        deltas[delta] += 1
        counts['snapshot_race_retries'] += timing.get('snapshot_race_retries', 0)
        held = row['commitment_active']
        kind = 'held' if held else 'fresh'
        pipeline = None
        counts[kind] += 1
        action = row['action']
        cached = row.get('planner_cached')
        if (cached is not None) != bool(cached_mode):
            raise ValueError('Missing or unexpected cached planner record')
        applied = cached is not None and cached['applied']
        if (action != row['committed_action'] or action not in row['boundary_execution_allowed_actions']
                or ('applied_action' in timing and timing['applied_action'] != action)
                or (not held and not applied and action not in row['laser_risk_context']['candidates'])):
            raise ValueError('Baseline action or input annotation mismatch')
        counts['applied_action_samples'] += int('applied_action' in timing)
        if 'observation_age_seconds' in timing:
            age = timing['observation_age_seconds']
            times[f'{kind}_observation_to_apply'].append(age)
            if 'input_send_seconds' in timing:
                send_age = age + timing['input_send_seconds']
                times[f'{kind}_observation_to_send_complete'].append(send_age)
                if previous_info and previous_info['elapsed_frames'] == info['elapsed_frames'] - delta:
                    accepted_read = previous_info.get('timing', {}).get('sample_read_seconds')
                    if accepted_read is not None:
                        pipeline = accepted_read + send_age
                        times[f'{kind}_capture_to_send_complete'].append(pipeline)
                        counts[f'{kind}_capture_to_send_over_16_6ms'] += int(pipeline > .0166)
        previous_info = info
        if ('input_dispatch' in timing) != (input_samples is not None):
            raise ValueError('Missing or unexpected input dispatch profile')
        if input_samples is not None:
            input_samples.append(_dispatch_sample(row, pipeline, input_contract))
        if cached is not None:
            candidate = cached['candidate']
            switch = cached.get('switch_check')
            if (('switch_check' in cached) != bool(switch_guard)
                    or bool(switch) != bool(switch_guard and cached['reason'])
                    or (switch and switch['status'] != 'clear' and cached['eligible'])):
                raise ValueError('Switch guard annotation mismatch')
            if switch:
                counts[f"switch_{switch['status']}"] += 1
            expected = candidate['suggested_action'] if applied and candidate else cached['baseline_action']
            if (action != expected or (cached['baseline_hold_active'] and cached['search_attempted'])
                    or (candidate and (candidate['clearance_px'] <= CACHE_CONFIG['clearance_px']
                                       or candidate['prefix_frames'] < SHORT_HORIZON))
                    or cached['reason'] != intervention_reason(candidate)
                    or (cached['eligible'] and (not cached['same_frame'] or not cached['reason']))
                    or applied != (cached_mode == 'intervene' and cached['eligible'])):
                raise ValueError('Cached planner decision mismatch')
            if cached['worked']:
                times['cached_planner_work'].append(cached['work_seconds'])
            if cached['search_attempted']:
                times['cached_search_work'].append(cached['work_seconds'])
            if candidate and candidate['source'] == 'cache':
                times['cached_recheck_work'].append(cached['work_seconds'])
            counts['cached_eligible'] += int(cached['eligible'])
            counts['cached_applied'] += int(applied)
            counts['cached_held_interruptions'] += int(applied and cached['baseline_hold_active'])
            counts['cached_frame_changed'] += int(cached['same_frame'] is False)
            counts['cached_triggers'] += int(cached['reason'] is not None)
            stops[cached['stop_reason']] += 1
        shadow = row.get('planner_shadow')
        expects_shadow = bool(shadow_config) and (not held or not shadow_config['fresh_decisions_only'])
        if (shadow is not None) != expects_shadow:
            raise ValueError('Missing or unexpected planner shadow sample')
        if shadow is None:
            continue
        route = shadow['route_actions']
        proposal = shadow['suggested_action']
        prefix = shadow['checked_prefix_frames']
        if shadow_config in (BUFFERED_REUSE_CONFIG, BUFFERED_LEAF_CONFIG):
            source = shadow['source']
            durations = shadow['route_durations']
            status = shadow['recheck']
            if shadow_config == BUFFERED_LEAF_CONFIG:
                limit = shadow_config['cache_warning_lookahead_frames']
                warning_ttc = shadow['cache_warning_ttc_frames']
                if ((status == 'warning_ttc') != (warning_ttc is not None and 0 < warning_ttc < limit)
                        or (warning_ttc is not None and not 0 < warning_ttc <= limit)
                        or (status == 'warning_ttc' and source == 'hold' and len(route) != 1)):
                    raise ValueError('Buffered warning TTC annotation mismatch')
            if (len(route) != len(durations) or any(not 1 <= value <= HOLD for value in durations)
                    or prefix != sum(durations)
                    or (source == 'cache' and status != 'clear')
                    or (source == 'hold' and status != 'clear'
                        and not (held and shadow_config == BUFFERED_LEAF_CONFIG and status == 'warning_ttc'))
                    or (held and source == 'search' and status != 'collision')
                    or (held and source in ('cache', 'hold') and proposal != action)
                    or shadow['cache_reused'] != (source == 'cache')):
                raise ValueError('Buffered reuse annotation mismatch')
            skipped_inference = row.get('baseline_inference_skipped', False)
            if skipped_inference and not held:
                raise ValueError('Buffered controller skipped a fresh inference')
            counts['held_baseline_inference_skipped'] += int(skipped_inference)
            counts[f'{kind}_source_{source}'] += 1
            counts[f'{kind}_prefix_at_least_16'] += int(prefix >= SHORT_HORIZON)
            counts[f'{kind}_scored_leaf_ttc'] += int(shadow['leaf_ttc_frames'] is not None)
            counts[f'{kind}_scored_time_limit_ttc'] += int(
                shadow['stop_reason'] == 'time_limit' and shadow['leaf_ttc_frames'] is not None)
            counts[f'{kind}_recheck_{shadow["recheck"]}'] += 1
            times[f'{kind}_baseline_controller'].append(shadow['baseline_seconds'])
            times[f'{kind}_shadow_work'].append(shadow['shadow_seconds'])
            if shadow['recheck'] != 'not_run':
                times['buffered_recheck'].append(shadow['recheck_seconds'])
            if source == 'cache':
                times['buffered_reuse_work'].append(shadow['shadow_seconds'])
                if pipeline is not None:
                    times['buffered_reuse_capture_to_send_complete'].append(pipeline)
                    counts['buffered_reuse_capture_to_send_over_10ms'] += int(pipeline > .010)
            if source == 'search':
                times['buffered_search'].append(shadow['search_seconds'])
            if 'policy_logits_seconds' in row and not skipped_inference:
                times['policy_logits'].append(row['policy_logits_seconds'])
                times['baseline_without_logits'].append(shadow['baseline_seconds'] - row['policy_logits_seconds'])
            for name, value in (row.get('previous_trace_timing') or {}).items():
                times[f'trace_{name.removesuffix("_seconds")}'].append(value)
        elif prefix != HOLD * len(route):
            raise ValueError('Shadow route prefix mismatch')
        if (shadow['baseline_action'] != action or proposal != (route[0] if route else None)
                or shadow['checked_edges'] > shadow_config['budget']['max_edges']):
            raise ValueError('Shadow action or budget annotation mismatch')
        counts['shadow_samples'] += 1
        counts['long_requests'] += int(shadow['horizon_frames'] == LONG_HORIZON)
        counts['changed_suggestions'] += int(proposal is not None and proposal != action)
        counts['no_suggestion'] += int(proposal is None)
        counts['suggestions_outside_execution_bounds'] += int(
            proposal is not None and proposal not in row['boundary_execution_allowed_actions'])
        counts['baseline_hold_rejected_with_alternative'] += int(
            shadow.get('baseline_first_hold_safe') is False and proposal is not None)
        stops[shadow['stop_reason']] += 1
        if 'planner_seconds' in shadow:
            counts['planner_over_budget_ms'] += int(
                shadow['planner_seconds'] * 1000 > shadow_config['budget']['max_ms'])
            times['planner'].append(shadow['planner_seconds'])
        times['shadow_decode_and_plan'].append(shadow['shadow_seconds'])
    return {'outcome': end, 'counts': dict(counts), 'stop_reasons': dict(stops), 'delta_frames': dict(deltas),
            'timing_ms': {name: timing_ms(values) for name, values in times.items()},
            'first_partition': first_partition(partition_samples)}


def analyze(path):
    rows, ends, errors = defaultdict(list), {}, []
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        raw = stream.readline()
        digest.update(raw)
        header = json.loads(raw)
        if header.get('schema') != LASER_SCHEMA or header.get('task') != LASER_TASK:
            raise ValueError('Expected a Young Demon Lord laser evaluation log')
        shadow_config = header.get('planner_shadow')
        if shadow_config and shadow_config not in (SHADOW_CONFIG, BUFFERED_TIMING_CONFIG,
                                                    BUFFERED_REUSE_CONFIG, BUFFERED_LEAF_CONFIG):
            raise ValueError('Unsupported planner shadow configuration')
        input_contract = header.get('input_timing_contract')
        if input_contract is not None and (input_contract not in (INPUT_TIMING_CONTRACT, MULTI_KEY_INPUT_TIMING_CONTRACT)
                                           or shadow_config != BUFFERED_LEAF_CONFIG):
            raise ValueError('Unsupported input timing contract')
        cached_config = header.get('planner_cache_config')
        cached_mode = header.get('planner_cache_mode')
        switch_guard = header.get('planner_switch_guard')
        if cached_mode is not None and (cached_config != CACHE_CONFIG or cached_mode not in ('shadow', 'intervene')
                                        or header.get('hold_frames') != HOLD or header.get('repeat') != 2):
            raise ValueError('Unsupported cached planner configuration')
        if switch_guard is not None and (cached_mode is None or switch_guard != SWITCH_GUARD):
            raise ValueError('Unsupported switch guard configuration')
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row['event'] == 'transition':
                del row['next_observation']
                obs = row['observation']
                row['observation'] = {key: obs[key] for key in ('player', 'lasers')}
                rows[row['episode']].append(row)
            elif row['event'] == 'reset':
                rows[row['episode']]
            elif row['event'] == 'episode_end':
                ends[row['episode']] = {key: row[key] for key in ('reason', 'elapsed_frames', 'deaths')}
            elif row['event'] == 'error':
                errors.append(row)
    slack = header.get('candidate_slack', header.get('risk_filter', {}).get('max_clearance_loss'))
    if cached_mode:
        work = [cached['work_seconds'] for samples in rows.values() for row in samples
                if (cached := row.get('planner_cached')) and cached['worked']]
        fresh_work = [cached['work_seconds'] for samples in rows.values() for row in samples
                      if (cached := row.get('planner_cached')) and cached['worked'] and not cached['baseline_hold_active']]
        cache_summary = {'mode': cached_mode, 'config': cached_config, 'work_ms': timing_ms(work),
                         'fresh_work_ms': timing_ms(fresh_work),
                         'build_md5': header.get('build_md5')}
    else:
        cache_summary = None
    input_samples = [] if input_contract else None
    episodes = {episode: summarize_episode(samples, ends.get(episode), shadow_config, cached_mode,
                                            switch_guard, input_samples, input_contract)
                for episode, samples in rows.items()}
    result = {'source': str(path.resolve()), 'sha256': digest.hexdigest(), 'model_sha256': header['model_sha256'],
            'candidate_slack': slack, 'planner_shadow': shadow_config, 'errors': errors,
            'planner_cache': cache_summary, 'planner_switch_guard': switch_guard,
            'episodes': episodes}
    if input_samples is not None:
        result['input_timing'] = _input_timing_report(input_samples, input_contract)
    return result


def validate_cached_shadow(path, model_hash, build, candidate_slack, *, switch_guard=False):
    report = analyze(path)
    cache = report['planner_cache']
    episodes = list(report['episodes'].values())
    if (cache is None or cache['mode'] != 'shadow' or cache['build_md5'] != build
            or report['model_sha256'] != model_hash or report['candidate_slack'] != candidate_slack
            or bool(report.get('planner_switch_guard')) != switch_guard
            or report['errors'] or len(episodes) != 10
            or any(not ep['outcome'] or ep['outcome']['reason'] not in ('spell_success', 'spell_failure')
                   or not ep['counts'].get('fresh', 0)
                   for ep in episodes)
            or not any(ep['counts'].get('cached_eligible', 0) for ep in episodes)
            or cache['work_ms'].get('p95', float('inf')) >= 5.
            or cache['fresh_work_ms'].get('p95', float('inf')) >= 5.):
        raise ValueError('Intervention requires ten matching completed shadow episodes, eligible suggestions, and p95 below 5 ms')
    return report['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(analyze(args.trace), indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report)
    print(report, end='')


if __name__ == '__main__':
    main()
