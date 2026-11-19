"""Evaluate causal trigger coverage on recorded contraction diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def activations(rows, mode, threshold):
    active = []
    previous = None
    for row in rows:
        count = row['layers'][-1]['sequences']
        if row['baseline_request'] is not None:
            if mode == 'danger':
                trigger = row['danger']
            else:
                current = count if mode == 'sequences' else len(row['planner_directions'])
                trigger = previous is not None and previous > 0 and current <= previous * (1 - threshold)
                previous = current
            if trigger:
                active.append(row['frame'])
    return active


def evaluate(report, mode, threshold):
    cases = []
    for case in report['cases']:
        scans = case['scan']
        failure = scans[case['failed_policy']]
        control = scans[case['comparison_policy']]
        frames = activations(failure, mode, threshold)
        control_frames = activations(control, mode, threshold)
        known = [row['frame'] for row in case['full_trace']
                 if row['outcome'] == 'trace_end' and row['first_below_padding_frame'] is None]
        trials = [row for row in case['trials'] if row['frame'] in frames]
        successes = [row for row in trials if row['through_frame'] == case['checkpoint']
                     and row['outcome'] in ('window_end', 'trace_end')
                     and row['first_below_padding_frame'] is None]
        cases.append({'source': case['source'], 'delay': case['delay'],
                      'failed_frames': frames, 'successful_control_frames': control_frames,
                      'known_full_trace_escape_frames': known,
                      'covered_escape_frames': sorted(set(known) & set(frames)),
                      'available_trials_at_triggers': len(trials),
                      'safe_checkpoint_trials_at_triggers': len(successes),
                      'first_trigger_lead_frames': case['stop'] - frames[0] if frames else None,
                      'failed_decisions': sum(row['baseline_request'] is not None for row in failure),
                      'control_decisions': sum(row['baseline_request'] is not None for row in control)})
    return {'mode': mode, 'drop_fraction': threshold, 'cases': cases,
            'covered_cases': sum(bool(c['covered_escape_frames']) for c in cases),
            'failed_activations': sum(len(c['failed_frames']) for c in cases),
            'control_activations': sum(len(c['successful_control_frames']) for c in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    raw = args.input.read_bytes()
    source = json.loads(raw)
    variants = [evaluate(source, 'danger', None)]
    for mode in ('sequences', 'planner_directions'):
        for threshold in (.2, .5, .8):
            variants.append(evaluate(source, mode, threshold))
    result = {'source': str(args.input), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'limitations': ['Retrospective diagnostic windows selected around known failures.',
                              'Successful-control activations are not labeled false positives.',
                              'Trial coverage does not validate action selection or closed-loop triggering.',
                              'Sequence counts require exhaustive search; latency is unmeasured.'],
              'variants': variants}
    for variant in variants:
        print(variant['mode'], variant['drop_fraction'], 'covered', variant['covered_cases'],
              'failure/control activations', variant['failed_activations'], variant['control_activations'])
    if args.input.read_bytes() != raw:
        raise AssertionError('Input changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
