"""Compare first-move swaps and longer windows on training captures."""
import argparse
import json
from pathlib import Path

import torch

from agent.replay_hrl import HOLD, ReplayWorker
from scripts.audit_training_escape import fork_env
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, collect, finish, restore, summarize

LATE_FRAMES = (384, 512, 640, 768, 896)
WINDOW = 96


def predict(policy, obs):
    return int(policy.predict(obs, deterministic=True)[0])


def first_move(env, slot):
    return env.choices[slot].route.actions[0] if env.choices else None


def collect_late(env, late):
    if (late is not None and env.frame in LATE_FRAMES and not env.danger
            and env.choices and env.frame + WINDOW <= env.episode.end):
        late.append(fork_env(env))


def rollout(env, obs, policy, first_slot=None, late=None):
    while True:
        collect_late(env, late)
        slot = predict(policy, obs) if first_slot is None else first_slot
        first_slot = None
        obs, _, done, _, info = env.step(slot)
        if done:
            return {'source': env.episode.source, 'episode': env.episode.number,
                    'delay': env.fixed_delay, **info}


def compare(episodes, worker, policies, index, delay, late):
    old, trained = policies
    env = CoverageEnv(episodes, worker, verify=True)
    obs, _ = env.reset(seed=0, options={'episode': index, 'delay': delay})
    first_slot_change = None
    while True:
        old_slot, trained_slot = predict(old, obs), predict(trained, obs)
        if first_slot_change is None and env.choices and old_slot != trained_slot:
            first_slot_change = env.frame
        old_move, trained_move = first_move(env, old_slot), first_move(env, trained_slot)
        if old_move != trained_move:
            frame = env.frame
            divergence = {'frame': frame, 'hold': HOLD, 'position': env.position.tolist(), 'danger': env.danger,
                          'old_slot': old_slot, 'trained_slot': trained_slot,
                          'old_move': old_move, 'trained_move': trained_move,
                          'old_leaf_ttc': env.choices[old_slot].ttc,
                          'trained_leaf_ttc': env.choices[trained_slot].ttc}
            # Swaps share the exact position, input queue, and hazard observation.
            results = {'old': rollout(fork_env(env), obs, old, old_slot, late),
                       'trained': rollout(fork_env(env), obs, trained, trained_slot),
                       'trained_first_old_tail': rollout(fork_env(env), obs, old, trained_slot),
                       'old_first_trained_tail': rollout(env, obs, trained, old_slot)}
            return {'first_slot_change': first_slot_change, 'divergence': divergence, 'results': results}
        collect_late(env, late)
        obs, _, done, _, info = env.step(old_slot)
        if done:
            row = {'source': env.episode.source, 'episode': env.episode.number, 'delay': delay, **info}
            return {'first_slot_change': first_slot_change, 'divergence': None,
                    'results': {'old': row, 'trained': row}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    sources = {str(path): digest(path) for path in [BASE, *TRAIN, args.old, args.trained]}
    episodes = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    backend.model.policy.requires_grad_(False).eval()
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policies = [PPO.load(path, device='cpu') for path in (args.old, args.trained)]
    report = {'sources': sources, 'split': 'train', 'candidate_coverage': 'all_verified_waypoints',
              'late_frames': LATE_FRAMES, 'paired': [], 'windows': {}}
    late = []
    for index, episode in enumerate(episodes):
        for delay in (0, 1):
            row = compare(episodes, worker, policies, index, delay, late)
            report['paired'].append(row)
            print(episode.source, episode.number, delay,
                  'divergence', (row['divergence'] or {}).get('frame'),
                  {name: (r['frames'], r['outcome']) for name, r in row['results'].items()}, flush=True)
    pools = collect(episodes, worker, policies[0])
    groups = [('ordinary_late_96', late, WINDOW),
              ('escape_to_trace_end', pools['escape'], None),
              ('jitter_to_trace_end', pools['jitter'], None)]
    for name, snapshots, duration in groups:
        report['windows'][name] = {}
        for label, policy in zip(('old', 'trained'), policies):
            rows = [finish(*restore(s, duration or s.episode.end - s.frame), policy) for s in snapshots]
            result = summarize(episodes, rows)
            report['windows'][name][label] = result
            print(name, label, result['safe_completed'], len(rows), result['mean_frames'], flush=True)
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise AssertionError('Source changed during diagnostics')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
