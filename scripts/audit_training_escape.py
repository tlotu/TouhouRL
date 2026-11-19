"""Audit one-hold alternatives from failed training PPO trajectories."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch

from agent.replay_hrl import ReplayManagerEnv, ReplayWorker, rebase
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture


def fork_env(env):
    """Copy mutable rollout state while sharing read-only captures and weights."""
    shared = [env.worker, env.episodes, *env.episodes]
    return deepcopy(env, {id(value): value for value in shared})


def continue_branch(origin, policy, choice=None):
    env = fork_env(origin)
    sample = env.episode.samples[env.frame]
    obs = env._observation(rebase(sample.observation, env.position), sample)
    first = True
    while True:
        if first and choice is not None:
            env.choices = (choice,)
            slot = 0
        else:
            slot = int(policy.predict(obs, deterministic=True)[0])
        obs, _, done, _, info = env.step(slot)
        first = False
        if done:
            break
    return {'through_frame': env.frame, 'outcome': info['outcome'],
            'min_clearance_px': info['min_clearance_px'],
            'first_below_padding_frame': info['first_below_padding_frame'],
            'history': [row for row in info['history'] if row['frame'] >= origin.frame]}


def physical_trace(history):
    return [(row['frame'], row['request'], row['end']) for row in history]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offsets', type=int, nargs='+', default=[120, 60, 32, 16])
    parser.add_argument('--episode', type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    if any(offset <= 0 or offset % 4 for offset in args.offsets):
        parser.error('--offsets must be positive multiples of four')
    torch.set_num_threads(1)
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, args.old, args.baseline]}
    previous = json.loads(args.baseline.read_text())
    if any(sources.get(path) != value for path, value in previous['sources'].items()):
        raise ValueError('Baseline source hashes do not match')
    episodes = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policy = PPO.load(args.old, device='cpu')
    report = {'sources': sources, 'split': 'train', 'offsets': sorted(set(args.offsets), reverse=True),
              'execution_hold': 4, 'verified_horizon': 16, 'padding_px': 2,
              'continuation': 'original_ppo', 'states': []}
    failures = [row for row in previous['evaluation']['episodes'] if row['outcome'] != 'trace_end']
    if args.episode is not None:
        failures = [row for row in failures if row['episode'] == args.episode]
        if not failures:
            parser.error('No failed training branch matches --episode')
    for failure in failures:
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (failure['source'], failure['episode']))
        stop = failure['start'] + failure['frames']
        frames = {stop - offset for offset in report['offsets']}
        env = ReplayManagerEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': failure['delay']})
        snapshots = []
        while True:
            if env.frame in frames:
                snapshots.append(fork_env(env))
            obs, _, done, _, info = env.step(int(policy.predict(obs, deterministic=True)[0]))
            if done:
                break
        if env.frame != stop or physical_trace(info['history']) != physical_trace(failure['history']):
            raise AssertionError('Baseline trajectory did not reproduce')
        for origin in snapshots:
            control = continue_branch(origin, policy)
            expected = [h for h in failure['history'] if h['frame'] >= origin.frame]
            if physical_trace(control['history']) != physical_trace(expected):
                raise AssertionError('Snapshot continuation did not reproduce')
            sample = origin.episode.samples[origin.frame]
            rebased = rebase(sample.observation, origin.position)
            choices, edges, ms = worker.options(origin.scene, rebased, origin.belief, by_direction=True)
            row = {'source': origin.episode.source, 'episode': origin.episode.number,
                   'delay': failure['delay'], 'origin': origin.frame, 'baseline_stop': stop,
                   'position': origin.position.tolist(), 'buffer': asdict(origin.buffer),
                   'belief': [asdict(q) for q in sorted(origin.belief, key=lambda q: (q.requested, q.applied, q.pending))],
                   'candidate_ms': ms, 'checked_edges': edges, 'control': control, 'actions': []}
            for action in range(9):
                candidate = {'action': action, 'verified': action in choices,
                             'original_slots': [i for i, c in enumerate(origin.choices)
                                                if c.route.actions[0] == action]}
                if action in choices:
                    result = continue_branch(origin, policy, choices[action])
                    candidate.update(leaf_ttc=choices[action].ttc, clearance=choices[action].gap,
                                     crossed_stop_by_16=result['through_frame'] >= stop + 16,
                                     closed_loop=result)
                row['actions'].append(candidate)
            report['states'].append(row)
            print(origin.episode.number, failure['delay'], origin.frame,
                  'verified', sorted(choices), 'past_stop',
                  [c['action'] for c in row['actions'] if c.get('crossed_stop_by_16')],
                  'trace_end', [c['action'] for c in row['actions']
                                if c.get('closed_loop', {}).get('outcome') == 'trace_end'], flush=True)
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise AssertionError('Source changed during audit')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
