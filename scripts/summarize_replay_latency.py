"""Locate tail decisions and attribute their wall-clock time to replay phases."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.compare_replay_reselection import key, timing


def decisions(case, reference):
    row, old = case['controller'], reference['controller']
    for field in ('history', 'attack_position', 'outcome', 'min_clearance_px', 'through_frame',
                  'route_count_decisions'):
        if row[field] != old[field]:
            raise ValueError(f'{key(case)} differs in {field}')

    def behavior(events):
        return [{k: v for k, v in event.items() if not k.endswith('_ms')} for event in events]

    if behavior(row['events']) != behavior(old['events']):
        raise ValueError(f'{key(case)} has different reselection events')
    events = {e['frame']: e for e in row['events']}
    for phase, total, history, earlier_ms in zip(
            row['decision_phases'], row['extra_decision_ms'], row['history'], old['extra_decision_ms'], strict=True):
        frame = phase['frame']
        if frame != history['frame']:
            raise ValueError(f'{key(case)} has a mismatched decision frame')
        event = events.get(frame, {})
        ms = phase['ms']
        measured = sum(ms.get(k, 0.) for k in ('exact_count', 'primitive', 'forecast_selection'))
        if measured > total:
            raise ValueError(f'{key(case)} has overlapping top-level timings')
        if event.get('forecast_calls', 0):
            activity = 'forecast'
        elif phase['calls'].get('exact_count', 0):
            activity = 'count_without_forecast'
        else:
            activity = 'no_count_or_forecast'
        yield {'case': key(case), 'frame': frame, 'total_ms': total, 'reference_ms': earlier_ms,
               'activity': activity, 'event': bool(event), 'forecast_calls': event.get('forecast_calls', 0),
               'options_other_ms': ms.get('options', 0.) - sum(ms.get(k, 0.) for k in ('plan', 'forced_plan', 'verify')),
               **phase}


def summarize(rows):
    ordered = sorted(rows, key=lambda r: r['total_ms'], reverse=True)
    distribution = timing([r['total_ms'] for r in ordered])
    cutoff = distribution['p95']
    tail = [r for r in ordered if r['total_ms'] >= cutoff]
    groups = {}
    for row in ordered:
        groups.setdefault(row['activity'], []).append(row)
    return {'decisions': len(rows), 'decision_ms': distribution,
            'by_activity': {name: {'decisions': len(group),
                                   'decision_ms': timing([r['total_ms'] for r in group])}
                            for name, group in groups.items()},
            'phase_total_ms': {phase: sum(r['ms'].get(phase, 0.) for r in ordered)
                               for phase in ('exact_count', 'primitive', 'forecast_selection', 'options',
                                             'plan', 'forced_plan', 'verify')},
            'tail_activity': dict(Counter(r['activity'] for r in tail)),
            'tail_count_calls': dict(Counter(r['calls'].get('exact_count', 0) for r in tail)),
            'near_p95': sorted(ordered, key=lambda r: abs(r['total_ms'] - cutoff))[:10],
            'tail': tail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    references = {}
    for path in args.reference:
        shard = json.loads(path.read_text(encoding='utf-8'))
        if shard['sources'] != profile['sources'] or shard['method'] != profile['method']:
            raise ValueError(f'{path} has different inputs or method')
        for case in shard['cases']:
            case_key = key(case)
            if case_key in references:
                raise ValueError(f'Duplicate case {case_key}')
            references[case_key] = case
    if set(references) != {key(c) for c in profile['cases']}:
        raise ValueError('Case mismatch')
    rows = [r for c in profile['cases'] for r in decisions(c, references[key(c)])]
    parallel_p95 = float(np.percentile([r['reference_ms'] for r in rows], 95))
    parallel_tail = [r for r in rows if r['reference_ms'] >= parallel_p95]
    result = {'profile': str(args.profile), 'reference': [str(p) for p in args.reference],
              'identical_histories_and_events': True, 'reference_p95_ms': parallel_p95,
              'reference_tail_activity': dict(Counter(r['activity'] for r in parallel_tail)),
              **summarize(rows)}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
