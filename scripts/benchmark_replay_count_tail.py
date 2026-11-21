"""Compare exact route enumeration on reconstructed count-heavy replay states."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from agent.replay_hrl import HOLD, PADDING, ReplayWorker, hold_outcomes
from agent.swept_collision import SweptCollision
from scripts.audit_replay_terminal_routes import feasible_prefixes
from scripts.compare_replay_reselection import key
from scripts.diagnose_replay_mixed_windows import predict
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def measure(scene, belief, *, prune=False, reuse=False, counts=True, vector=False, cull_batches=False):
    start = perf_counter()
    for _, layer in feasible_prefixes(
            scene, belief, (0, 1), PADDING, final_counts=counts, batch_size=1024,
            model_factory=lambda s: SweptCollision(s, clearance_cap=1e-6, cull_batches=cull_batches),
            prune_failed=prune, reuse_edges=reuse, vector_terminal=vector):
        pass
    ms = (perf_counter() - start) * 1000
    if not counts:
        by_first = Counter(actions[0] for actions, _ in layer)
        layer = [by_first[a] for a in range(9)]
    return layer, ms


def states(env, case, target):
    history = case['controller']['history']
    np.testing.assert_allclose(env.position, history[0]['start'], atol=1e-7, rtol=0)
    belief = env.belief
    for hold in history:
        frame = hold['frame']
        if frame > target:
            break
        scene = replace(env.episode.samples[frame].scene, player=tuple(hold['start']))
        if frame >= target - HOLD:
            yield frame, scene, belief
        outcomes = hold_outcomes(scene, belief, hold['request'])
        belief = frozenset(outcomes[tuple(np.round(hold['end'], 6))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--tail', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--vector', action='store_true', help='Compare the terminal-layer implementations.')
    modes.add_argument('--batch-culling', action='store_true', help='Compare collision batch culling.')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    tail = json.loads(args.tail.read_text(encoding='utf-8'))
    targets = [r for r in tail['near_p95'] if r['activity'] == 'count_without_forecast'][:3]
    torch.set_num_threads(1)
    episodes = [ep for path in TRAIN for ep in load_capture(path, digest(BASE))]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    old = PPO.load(Path('runs/hrl-young-replay-smoke-001/manager.zip'), device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    if args.batch_culling:
        variants = [('vector', dict(prune=True, reuse=True, vector=True)),
                    ('culled', dict(prune=True, reuse=True, vector=True, cull_batches=True))]
    elif args.vector:
        variants = [('reused', dict(prune=True, reuse=True)),
                    ('vector', dict(prune=True, reuse=True, vector=True))]
    else:
        variants = [('unpruned', {}), ('pruned', dict(prune=True)), ('reused', dict(prune=True, reuse=True))]
    rows = []
    for target in targets:
        case = next(c for c in profile['cases'] if list(key(c)) == target['case'])
        source, number, _, _, anchor = key(case)
        index = next(i for i, ep in enumerate(episodes) if (ep.source, ep.number) == (source, number))
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor:
            obs, _, done, _, _ = env.step(predict(old, obs))
            if done:
                raise ValueError('Control ended before anchor')
        for frame, scene, belief in states(env, case, target['frame']):
            expected, _ = measure(scene, belief, counts=False)
            for repeat in range(3):
                order = variants if repeat % 2 == 0 else reversed(variants)
                for name, options in order:
                    counts, ms = measure(scene, belief, **options)
                    if counts != expected:
                        raise AssertionError(f'{key(case)} frame {frame}: {name} count differs')
                    rows.append(dict(case=key(case), frame=frame, repeat=repeat, variant=name,
                                     counts=counts, ms=ms))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
