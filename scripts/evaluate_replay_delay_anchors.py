"""Evaluate policy choices and closed-loop outcomes at audited delay anchors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.replay_hrl import ReplayWorker
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, finish, paired_summary, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    paths = {'old': args.old, 'independent': args.independent, 'paired': args.paired}
    sources = {str(path): digest(path) for path in [BASE, *TRAIN, args.audit, *paths.values()]}
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    if any(audit['sources'].get(str(path)) != sources[str(path)] for path in [BASE, *TRAIN, args.old]):
        raise ValueError('Audit sources do not match')
    episodes = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    backend.model.policy.requires_grad_(False).eval()
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policies = {name: PPO.load(path, device='cpu') for name, path in paths.items()}
    anchors = []
    for pair in audit['pairs']:
        states = sorted((row for row in audit['states']
                         if row['source'] == pair['source'] and row['frame'] == pair['frame']),
                        key=lambda row: row['delay'])
        first, second = states
        if not all(pair[key] for key in ('observation_identical', 'actual_buffer_identical', 'belief_identical')):
            raise ValueError('Audit states do not share the same starting state')
        safe_moves = [{a['action'] for a in state['actions'] if a['outcome'] == 'trace_end'
                       and a['first_below_padding_frame'] is None} for state in (first, second)]
        common = safe_moves[0] & safe_moves[1]
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (first['source'], first['episode']))
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < first['frame']:
            obs, _, done, _, _ = env.step(int(policies['old'].predict(obs, deterministic=True)[0]))
            if done:
                raise AssertionError('Baseline stopped before anchor')
        np.testing.assert_array_equal(obs, first['observation'])
        moves = [choice.route.actions[0] for choice in env.choices]
        anchor = {'source': first['source'], 'episode': first['episode'], 'frame': env.frame,
                  'slot_moves': moves, 'common_safe_moves_with_old_tail': sorted(common), 'policies': {}}
        for label, policy in policies.items():
            tensor, _ = policy.policy.obs_to_tensor(obs)
            with torch.no_grad():
                probs = policy.policy.get_distribution(tensor).distribution.probs[0].cpu().numpy()
            rows = []
            for delay in (0, 1):
                branch, branch_obs = restore(env, duration=None)
                np.testing.assert_array_equal(branch_obs, obs)
                branch.fixed_delay = delay
                rows.append(finish(branch, branch_obs, policy))
            anchor['policies'][label] = {
                'first_move': moves[int(probs.argmax())],
                'common_safe_probability': sum(float(prob) for move, prob in zip(moves, probs) if move in common),
                'paired': paired_summary(rows), 'episodes': rows}
            print(env.frame, label, anchor['policies'][label]['paired'], flush=True)
        anchors.append(anchor)
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'anchors': anchors}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
