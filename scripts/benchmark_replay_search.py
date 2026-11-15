"""Compare full search and early exit on matched replay states and rollouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.replay_hrl import ReplayManagerEnv, ReplayWorker, rebase
from scripts.train_replay_hrl import BASE, TEST, PPO, PPOBackend, attack_position_summary, digest, evaluate, load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manager', type=Path, required=True)
    parser.add_argument('--prune-failed', action='store_true',
                        help='Compare failed-route pruning instead of early exit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    episodes = load_capture(TEST, digest(BASE))
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    manager = PPO.load(args.manager, device='cpu')
    full = ReplayWorker(backend.action_logits)
    fast = ReplayWorker(backend.action_logits, early_exit=not args.prune_failed, prune_failed=args.prune_failed)
    times = {name: [] for name in ('full', 'fast', 'full_exit', 'fast_exit')}
    counts = dict(states=0, early_exits=0, rule_changes=0, missing_first_moves=0, extra_first_moves=0)
    cases = [dict(episode=i, delay=d) for i in range(len(episodes)) for d in (0, 1)]
    env = ReplayManagerEnv(episodes, full)
    for case in cases:
        env.reset(seed=0, options=case)
        while True:
            obs = rebase(env.episode.samples[env.frame].observation, env.position)
            orders = (('full', full), ('fast', fast))
            if counts['states'] % 2:
                orders = tuple(reversed(orders))
            results = {name: worker.options(env.scene, obs, env.belief) for name, worker in orders}
            baseline, _, baseline_ms = results['full']
            alternative, edges, fast_ms = results['fast']
            if args.prune_failed:
                signature = lambda choices: [(c.route, c.gap, c.ttc) for c in choices]
                if signature(baseline) != signature(alternative):
                    raise AssertionError('Pruning changed the verified options')
            counts['states'] += 1
            times['full'].append(baseline_ms)
            times['fast'].append(fast_ms)
            if alternative and not edges:
                counts['early_exits'] += 1
                times['full_exit'].append(baseline_ms)
                times['fast_exit'].append(fast_ms)
            before = {c.route.actions[0] for c in baseline}
            after = {c.route.actions[0] for c in alternative}
            counts['missing_first_moves'] += len(before - after)
            counts['extra_first_moves'] += len(after - before)
            first = lambda choices: choices[0].route.actions[0] if choices else None
            counts['rule_changes'] += int(first(baseline) != first(alternative))
            _, _, done, _, _ = env.step(0)
            if done:
                break
    report = {'capture_sha256': digest(TEST), 'manager_sha256': digest(args.manager),
              'variant': 'prune_failed' if args.prune_failed else 'early_exit', **counts,
              'matched_timing': {name: {'median_ms': float(np.median(values)),
                                        'p95_ms': float(np.percentile(values, 95)),
                                        'over_16_6ms': sum(x > 16.6 for x in values)}
                                 for name, values in times.items() if values},
              'rollouts': {}}
    print(report['matched_timing'], counts, flush=True)
    for name, worker, policy in (
        ('fast_rule', fast, lambda _: 0),
        ('fast_manager', fast, lambda obs: manager.predict(obs, deterministic=True)[0].item()),
    ):
        result = evaluate(episodes, worker, policy, cases)
        result['attack_position'] = attack_position_summary(episodes, result['episodes'])
        report['rollouts'][name] = result
        print(name, result['outcomes'], result['mean_frames'], flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
