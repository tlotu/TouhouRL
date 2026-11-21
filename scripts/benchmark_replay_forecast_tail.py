"""Compare forecast reuse at the slowest recorded reselection decisions."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from agent.replay_hrl import HOLD, ReplayWorker
from agent.swept_collision import SweptCollision, UniqueSweeps
from scripts.audit_replay_replanning_scores import forecast, forecast_candidates
from scripts.benchmark_replay_count_tail import states
from scripts.compare_replay_reselection import key
from scripts.diagnose_replay_mixed_windows import predict
from scripts.evaluate_replay_reselection import choose
from scripts.profile_replay_options import PhaseTimes
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def measure(origin, actions, nominal, horizon, policy, variant):
    factory = UniqueSweeps if variant in ('sweeps', 'bounded', 'ceiling') else SweptCollision
    worker = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True, collision_factory=factory)
    timer = PhaseTimes()
    pruned = {} if variant == 'ceiling' else None
    with timer.options(worker):
        started = perf_counter()
        if variant == 'independent':
            results = {a: forecast(origin, a, worker, horizon, policy) for a in actions}
        elif variant in ('bounded', 'ceiling'):
            results = choose(origin, policy, worker, nominal, horizon, actions, stop_at_horizon=True,
                             pruned=pruned)
        else:
            results = dict(forecast_candidates(origin, actions, worker, horizon, policy))
        elapsed = (perf_counter() - started) * 1000
    return results, {'total_ms': elapsed, 'pruned_forecasts': pruned or {}, **timer.take()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--tail', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--bound', action='store_true', help='Time the early exit at the single slowest event.')
    modes.add_argument('--multi', action='store_true', help='Record branch results for every multi-candidate event.')
    modes.add_argument('--ceiling', action='store_true', help='Time branch pruning at the slowest multi-candidate event.')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    tail = json.loads(args.tail.read_text(encoding='utf-8'))
    if args.multi or args.ceiling:
        targets = [row for row in tail['tail'] if row['forecast_calls'] > 1]
        if args.ceiling:
            targets = targets[:1]
    else:
        targets = [row for row in tail['tail'] if row['forecast_calls'] > 0][:1 if args.bound else 3]
    torch.set_num_threads(1)
    episodes = [ep for path in TRAIN for ep in load_capture(path, digest(BASE))]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    paths = {'old': 'hrl-young-replay-smoke-001', 'paired': 'hrl-young-delay-paired-001'}
    policies = {name: PPO.load(Path('runs') / directory / 'manager.zip', device='cpu')
                for name, directory in paths.items()}
    actual = ReplayWorker(backend.action_logits, prune_failed=True)
    rows = []
    if args.multi:
        variants = ('sweeps',)
    elif args.ceiling:
        variants = ('bounded', 'ceiling')
    elif args.bound:
        variants = ('sweeps', 'bounded')
    else:
        variants = ('independent', 'shared', 'sweeps')
    for target in targets:
        case = next(c for c in profile['cases'] if list(key(c)) == target['case'])
        source, episode, policy, _, anchor = key(case)
        index = next(i for i, ep in enumerate(episodes) if (ep.source, ep.number) == (source, episode))
        env = CoverageEnv(episodes, actual, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise ValueError('Control ended before anchor')
        frame, scene, belief = list(states(env, case, target['frame']))[-1]
        env.frame, env.scene, env.belief = frame, scene, belief
        event = next(e for e in case['controller']['events'] if e['frame'] == frame)
        horizon = 60 - HOLD if event['post_hold'] else 60
        expected = None
        for repeat in range(1 if args.multi else 3):
            for variant in variants if repeat % 2 == 0 else reversed(variants):
                result, row = measure(env, event['shortlisted_actions'], event['nominal'],
                                      horizon, policies[policy], variant)
                if variant in ('bounded', 'ceiling'):
                    selected, scores, calls = result
                    if selected != event['selected'] or any(event['mean_frames'][str(a)] != s
                                                            for a, s in scores.items()):
                        raise AssertionError('Bounded forecast differs from recorded event')
                    rows.append(dict(case=key(case), frame=frame, variant=variant, repeat=repeat,
                                     forecast_calls=calls, **row))
                    continue
                if expected is None:
                    expected = result
                if result != expected:
                    raise AssertionError(f'{key(case)} frame {frame}: {variant} forecasts differ')
                if {str(a): np.mean([b['frames'] for b in f['branches']]) for a, f in result.items()
                        if all(b['frames'] >= HOLD for b in f['branches'])} != event['mean_frames']:
                    raise AssertionError('Reconstructed scores differ from recorded event')
                rows.append(dict(case=key(case), frame=frame, variant=variant, repeat=repeat,
                                 forecast_calls=len(result), **row))
                if args.multi:
                    rows[-1].update(nominal=event['nominal'], selected=event['selected'],
                                    horizon=horizon, candidates=result)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
