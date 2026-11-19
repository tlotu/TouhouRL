"""Compare upper-policy observations and first-move outcomes across delays."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from agent.replay_hrl import ReplayWorker, rebase
from scripts.audit_training_escape import continue_branch
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    paths = [Path('runs/hrl-young-replay-smoke-001/manager.zip'),
             Path('runs/hrl-young-mixed-windows-001/manager.zip'),
             Path('runs/hrl-young-full-trace-001/manager.zip')]
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, *paths]}
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policies = [PPO.load(p, device='cpu') for p in paths]
    report = {'sources': sources, 'states': [], 'pairs': []}
    for source, frame in [(str(TRAIN[0]), 340), (str(TRAIN[1]), 344)]:
        index = next(i for i, ep in enumerate(episodes) if ep.source == source and ep.number == 2)
        pair = []
        for delay in (0, 1):
            env = CoverageEnv(episodes, worker, verify=True)
            obs, _ = env.reset(seed=0, options={'episode': index, 'delay': delay})
            history = []
            while env.frame < frame:
                history.append(obs.tolist())
                obs, _, done, _, _ = env.step(int(policies[0].predict(obs, deterministic=True)[0]))
                if done:
                    raise AssertionError('Baseline stopped before target')
            predictions = []
            for path, policy in zip(paths, policies):
                tensor, _ = policy.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    probabilities = policy.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
                slot = int(probabilities.argmax())
                predictions.append({'model': str(path), 'probabilities': probabilities.tolist(),
                                    'slot': slot, 'move': env.choices[slot].route.actions[0]})
            directions, _, _ = worker.options(env.scene, rebase(env.episode.samples[frame].observation,
                                                               env.position), env.belief, by_direction=True)
            row = {'source': source, 'episode': 2, 'delay': delay, 'frame': frame,
                   'observation': obs.tolist(), 'position': env.position.tolist(),
                   'actual_buffer': asdict(env.buffer),
                   'belief': [asdict(q) for q in sorted(env.belief, key=lambda q: (q.requested, q.applied, q.pending))],
                   'predictions': predictions, 'actions': []}
            for action, choice in sorted(directions.items()):
                result = continue_branch(env, policies[0], choice)
                row['actions'].append({'action': action, 'leaf_ttc': choice.ttc,
                                       'clearance': choice.gap, **result})
                print(source, delay, frame, action, result['through_frame'], result['outcome'], flush=True)
            report['states'].append(row)
            pair.append((obs, np.asarray(history), row))
        first, second = pair
        report['pairs'].append({'source': source, 'frame': frame,
            'observation_identical': bool(np.array_equal(first[0], second[0])),
            'max_observation_difference': float(np.max(np.abs(first[0] - second[0]))),
            'different_features': np.flatnonzero(first[0] != second[0]).tolist(),
            'history_different_frames': (np.flatnonzero(np.any(first[1] != second[1], axis=1)) * 4).tolist(),
            'actual_buffer_identical': first[2]['actual_buffer'] == second[2]['actual_buffer'],
            'belief_identical': first[2]['belief'] == second[2]['belief']})
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
