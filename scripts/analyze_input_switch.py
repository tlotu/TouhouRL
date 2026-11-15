"""Compare measured movement with the states in an ordered direction-key switch."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path

import numpy as np

from agent.input_switch import switch_states
from agent.short_planner import VECTORS


TOLERANCE = .02
LOWER, UPPER = (8., 16.), (376., 432.)


def classify_switch(start, end, frames, previous, requested):
    states = switch_states(previous, requested)
    matches = []
    for sequence in combinations_with_replacement(states, frames):
        position = np.asarray(start)
        for action in sequence:
            position = np.clip(position + 2 * VECTORS[action], LOWER, UPPER)
        if np.linalg.norm(position - end) <= TOLERANCE:
            matches.append(sequence)
    if (requested,) * frames in matches:
        result = 'nominal'
    elif any(set(seq) <= {previous, requested} for seq in matches):
        result = 'old_and_new_compatible'
    elif matches:
        result = 'intermediate_required_in_model'
    else:
        result = 'unexplained'
    return result, matches


def analyze(path):
    digest = hashlib.sha256()
    groups = defaultdict(Counter)
    brackets = defaultdict(Counter)
    details = []
    with path.open('rb') as stream:
        raw = next(stream)
        digest.update(raw)
        header = json.loads(raw)
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row['event'] != 'transition' or row['terminated'] or row['truncated']:
                continue
            info = row['info']
            timing = info['timing']
            previous, action = timing['previous_action'], timing['applied_action']
            frames = info['delta_frames']
            switch = row.get('switch_timing')
            dispatch = timing.get('input_dispatch')
            multi_key = dispatch is not None and len(dispatch['events']) > 1
            override = (row.get('planner_cached') or {}).get('applied')
            group = 'override' if override else 'probe' if switch is not None else 'baseline'
            if previous is None or not 1 <= frames <= 4:
                groups[group]['skipped'] += 1
                continue
            start, end = ((np.asarray(row[key]['player'][:2])+1)*[192,224]
                          for key in ('observation', 'next_observation'))
            result, matches = classify_switch(start, end, frames, previous, action)
            changed = previous != action
            if multi_key and changed:
                group = 'batched_multi_key' if 'batch' in dispatch else 'sequential_multi_key'
            groups[group if changed or override else 'steady'][result] += 1
            if override and not changed:
                groups[group]['unchanged_requested_direction'] += 1
            frame_start = (info['frame'] - frames) & 255
            bracket = None
            if switch is not None and changed:
                if switch['frame_before_send'] != frame_start:
                    bracket = 'advanced_before_send'
                elif switch['frame_after_send'] != frame_start:
                    bracket = 'advanced_during_send_bracket'
                else:
                    bracket = 'same_counter'
                brackets[bracket][result] += 1
            if (override or switch is not None or multi_key) and (changed or result != 'nominal'):
                details.append({'episode': row['episode']+1, 'frame': info['elapsed_frames']-frames,
                                'frames': frames, 'previous_action': previous, 'requested_action': action,
                                'displacement_px': (end-start).tolist(), 'classification': result,
                                'matching_sequences': matches, 'switch_bracket': bracket, 'group': group,
                                'observation_age_ms': 1000*timing['observation_age_seconds'],
                                'input_send_ms': 1000*timing['input_send_seconds']})
    return {'source': str(path.resolve()), 'sha256': digest.hexdigest(), 'build_md5': header.get('build_md5'),
            'model': 'Nominal 2 px/frame, observed frame count and field clipping. Matches describe displacement, not the observed order of engine input.',
            'groups': groups, 'switch_brackets': brackets, 'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.trace)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'groups': report['groups'], 'switch_brackets': report['switch_brackets']}, indent=2))
    print(f'Report: {args.output.resolve()}')


if __name__ == '__main__':
    main()
