"""Compare closed-loop reselection traces and measured forecast work."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_replay_hrl import digest


def key(case):
    row = case['controller']
    return row['source'], row['episode'], case['policy'], case['delay'], case['anchor']


def timing(values):
    return dict(zip(('p50', 'p95', 'max'), map(float, np.percentile(values, (50, 95, 100))))) if values else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    reference, candidate = [json.loads(p.read_text(encoding='utf-8')) for p in (args.reference, args.candidate)]
    if reference['sources'] != candidate['sources']:
        raise ValueError('Source mismatch')
    refs = {key(c): c for c in reference['cases']}
    if set(refs) != {key(c) for c in candidate['cases']}:
        raise ValueError('Case mismatch')
    cases, events, decisions = [], [], []
    for case in candidate['cases']:
        row, old = case['controller'], refs[key(case)]['controller']
        events.extend(row['events'])
        decisions.extend(row['extra_decision_ms'])
        cases.append({'key': key(case), 'outcome': row['outcome'], 'through_frame': row['through_frame'],
                      'min_clearance_px': row['min_clearance_px'],
                      'first_below_padding_frame': row['first_below_padding_frame'],
                      'identical_history': row['history'] == old['history'],
                      'identical_attack_position': row['attack_position'] == old['attack_position'],
                      'identical_selections': [(e['frame'], e['selected']) for e in row['events']] ==
                                              [(e['frame'], e['selected']) for e in old['events']]})
    all_calls = sum(len(e['offered_actions']) for e in events if len(e['offered_actions']) > 1)
    calls = sum(e['forecast_calls'] for e in events)
    result = {'inputs': {str(p): digest(p) for p in (args.reference, args.candidate)}, 'cases': cases,
              'events': len(events), 'forecast_calls': calls, 'all_offered_forecast_calls': all_calls,
              'forecast_reduction_fraction': 1 - calls / all_calls if all_calls else 0,
              'extra_decision_ms': timing(decisions), 'scoring_ms': timing([e['scoring_ms'] for e in events]),
              'selection_with_trigger_ms': timing([e['selection_ms'] for e in events]),
              'scope': 'Training capture replay; timing excludes normal environment step and game IO.'}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
