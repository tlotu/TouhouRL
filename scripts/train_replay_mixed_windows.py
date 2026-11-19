"""On-policy waypoint training with jittered escape and ordinary windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from agent.replay_hrl import ReplayManagerEnv, ReplayWorker
from scripts.audit_training_escape import fork_env
from scripts.train_replay_hrl import BASE, TRAIN, TEST, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_hrl import attack_position_summary


class CoverageEnv(ReplayManagerEnv):
    def _plan_options(self, obs):
        return self.worker.options(self.scene, obs, self.belief, waypoint_coverage=True)


def restore(snapshot, duration=96):
    env = fork_env(snapshot)
    env.__class__ = CoverageEnv
    env.start = env.frame
    env.limit = env.episode.end if duration is None else min(env.episode.end, env.frame + duration)
    env.times, env.edges, env.history = [], [], []
    env.min_gap = float('inf')
    env.padded_contact = env.last_padded_contact = None
    env.max_omitted = env.distinct_moves = 0
    env.attack_reward = True
    return env, env._prepare()


class MixedWindows(gym.Env):
    def __init__(self, escape, ordinary, duration=96, delay_sampling='recorded'):
        self.duration = duration
        self.delay_sampling = delay_sampling
        self.pending_pair = None
        self.delay_starts = Counter()
        self.pools = {'escape': escape, 'ordinary': ordinary}
        self.observation_space = escape[0].observation_space
        self.action_space = escape[0].action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.pending_pair = None
        if self.pending_pair is not None:
            self.group, snapshot, delay = self.pending_pair
            self.pending_pair = None
        else:
            self.group = 'escape' if self.np_random.random() < .5 else 'ordinary'
            pool = self.pools[self.group]
            snapshot = pool[int(self.np_random.integers(len(pool)))]
            delay = snapshot.fixed_delay
            if self.delay_sampling != 'recorded':
                delay = int(self.np_random.integers(2))
                if self.delay_sampling == 'paired':
                    self.pending_pair = self.group, snapshot, 1 - delay
        self.env, obs = restore(snapshot, self.duration)
        self.env.fixed_delay = delay
        self.delay_starts[self.group, delay] += 1
        return obs, {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Equal completion reward in both pools; no preferred direction label.
        bonus = float(done and info['outcome'] in ('window_end', 'trace_end')
                      and info['first_below_padding_frame'] is None)
        info['pool'] = self.group
        return obs, reward + bonus, done, truncated, info


class Stats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
        self.rollouts = []

    def _on_step(self):
        for info in self.locals['infos']:
            if info['outcome']:
                self.counts[info['pool'] + ':' + info['outcome']] += 1
        return True

    def _on_rollout_end(self):
        row = {'steps': self.num_timesteps, 'outcomes': dict(self.counts)}
        self.rollouts.append(row)
        print(row, flush=True)


def collect(episodes, worker, old):
    pools = {'escape': [], 'ordinary': [], 'jitter': []}
    for index, episode in enumerate(episodes):
        for delay in (0, 1):
            env = ReplayManagerEnv(episodes, worker, verify=True)
            obs, _ = env.reset(seed=0, options={'episode': index, 'delay': delay})
            targeted = episode.source == str(TRAIN[0]) and episode.number == 3
            center = 320 if delay == 0 else 288
            while True:
                if targeted and env.frame - center in (-12, -8, 0, 8, 12):
                    pools['escape'].append(fork_env(env))
                elif targeted and env.frame - center in (-4, 4):
                    pools['jitter'].append(fork_env(env))
                elif not targeted and env.frame in (128, 256) and not env.danger:
                    pools['ordinary'].append(fork_env(env))
                obs, _, done, _, _ = env.step(int(old.predict(obs, deterministic=True)[0]))
                if done:
                    break
    if any(not pool for pool in pools.values()):
        raise ValueError('Empty window pool')
    return pools


def collect_full(episodes, worker, old, diagnostic):
    pools = {'escape': [], 'ordinary': []}
    for case in diagnostic['paired']:
        row = case['results']['old']
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (row['source'], row['episode']))
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': row['delay']})
        pools['ordinary'].append(fork_env(env))
        if case['divergence'] is None:
            continue
        target = max(0, case['divergence']['frame'] - 16)
        while env.frame < target:
            obs, _, done, _, _ = env.step(int(old.predict(obs, deterministic=True)[0]))
            if done:
                raise ValueError('Baseline stopped before diagnostic origin')
        pools['escape'].append(fork_env(env))
    return pools


def finish(env, obs, policy):
    while True:
        obs, _, done, _, info = env.step(int(policy.predict(obs, deterministic=True)[0]))
        if done:
            return {'source': env.episode.source, 'episode': env.episode.number,
                    'delay': env.fixed_delay, **info}


def summarize(episodes, rows):
    return {'outcomes': dict(Counter(row['outcome'] for row in rows)),
            'safe_completed': sum(row['outcome'] in ('window_end', 'trace_end')
                                  and row['first_below_padding_frame'] is None for row in rows),
            'mean_frames': float(np.mean([row['frames'] for row in rows])),
            'below_padding': sum(row['first_below_padding_frame'] is not None for row in rows),
            'attack_position': attack_position_summary(episodes, rows), 'episodes': rows}


def paired_summary(rows):
    pairs = {}
    for row in rows:
        key = row['source'], row['episode'], row['start'], row.get('origin_delay')
        pair = pairs.setdefault(key, {})
        if row['delay'] in pair:
            raise ValueError(f'Duplicate delay at {key}')
        pair[row['delay']] = row
    both_safe = 0
    worst_frames = []
    for pair in pairs.values():
        if set(pair) != {0, 1}:
            raise ValueError('Missing paired delay')
        both_safe += all(row['outcome'] == 'trace_end' and row['first_below_padding_frame'] is None
                         for row in pair.values())
        worst_frames.append(min(row['frames'] for row in pair.values()))
    return {'both_safe': both_safe, 'pairs': len(pairs),
            'mean_worst_frames': float(np.mean(worst_frames))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-trace-diagnostic', type=Path)
    parser.add_argument('--delay-sampling', choices=('recorded', 'independent', 'paired'), default='recorded')
    parser.add_argument('--steps', type=int, default=4096)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 128:
        parser.error('--steps must be a positive multiple of 128')
    if args.delay_sampling != 'recorded' and not args.full_trace_diagnostic:
        parser.error('--delay-sampling requires --full-trace-diagnostic')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, TEST, args.old]}
    train = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    backend.model.policy.requires_grad_(False).eval()
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(args.old, device='cpu')
    duration = 96
    if args.full_trace_diagnostic:
        diagnostic = json.loads(args.full_trace_diagnostic.read_text())
        if any(sources.get(path, digest(Path(path))) != value
               for path, value in diagnostic['sources'].items()):
            raise ValueError('Diagnostic source hashes do not match')
        sources[str(args.full_trace_diagnostic)] = digest(args.full_trace_diagnostic)
        pools = collect_full(train, worker, old, diagnostic)
        duration = None
    else:
        pools = collect(train, worker, old)
    compare_delays = args.delay_sampling != 'recorded'
    if compare_delays:
        # Frame-zero snapshots are the same for both execution delays.
        pools['ordinary'] = [snapshot for snapshot in pools['ordinary'] if snapshot.fixed_delay == 0]
    report = {'sources': sources, 'seed': 7, 'steps': args.steps, 'window_frames': duration,
              'delay_sampling': args.delay_sampling,
              'escape_sampling_probability': .5, 'completion_bonus': 1.,
              'candidate_coverage': 'all_verified_waypoints', 'evaluations': {},
              'pools': {key: [{'source': e.episode.source, 'episode': e.episode.number,
                               'delay': e.fixed_delay, 'frame': e.frame} for e in pool]
                        for key, pool in pools.items()}}

    def save():
        temp = args.output / 'report.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(args.output / 'report.json')

    save()
    print({k: len(v) for k, v in pools.items()}, flush=True)
    env = MixedWindows(pools['escape'], pools['ordinary'], duration, args.delay_sampling)
    model = PPO.load(args.old, env=env, device='cpu', seed=7)
    stats = Stats()
    start = time.perf_counter()
    model.learn(args.steps, callback=stats)
    model.save(args.output / 'manager.zip')
    report.update(training_seconds=time.perf_counter() - start, rollouts=stats.rollouts,
                  reset_counts={f'{group}:{delay}': count for (group, delay), count in env.delay_starts.items()})
    save()
    test = load_capture(TEST, sources[str(BASE)])
    for label, policy in [('old', old), ('trained', model)]:
        report['evaluations'][label] = {}
        for name, pool in pools.items():
            rows = []
            for snapshot in pool:
                delays = (0, 1) if compare_delays else (snapshot.fixed_delay,)
                for delay in delays:
                    evaluation, obs = restore(snapshot, duration)
                    evaluation.fixed_delay = delay
                    row = finish(evaluation, obs, policy)
                    row['origin_delay'] = snapshot.fixed_delay
                    rows.append(row)
            result = summarize(train, rows)
            if compare_delays:
                result['paired'] = paired_summary(rows)
            report['evaluations'][label][name] = result
            print(label, name, result['outcomes'], result['mean_frames'], flush=True)
            save()
        rows = []
        for i in range(len(test)):
            for delay in (0, 1):
                evaluation = CoverageEnv(test, worker, verify=True)
                obs, _ = evaluation.reset(seed=0, options={'episode': i, 'delay': delay})
                rows.append(finish(evaluation, obs, policy))
        result = summarize(test, rows)
        if compare_delays:
            result['paired'] = paired_summary(rows)
        report['evaluations'][label]['held_out_full'] = result
        print(label, 'held_out_full', result['outcomes'], result['mean_frames'], flush=True)
        save()
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise AssertionError('Source changed')
    report['sources_unchanged'] = True
    save()


if __name__ == '__main__':
    main()
