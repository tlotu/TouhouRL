"""Train waypoint PPO from counterfactual prefixes of failed rule rollouts."""
import argparse
from copy import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from agent.replay_connectivity import forecast_horizon
from agent.replay_hrl import ReplayManagerEnv, ReplayWorker, rebase
from scripts.train_replay_hrl import BASE, DIAGNOSTICS, TEST, TRAIN, PPO, PPOBackend
from scripts.train_replay_hrl import attack_position_summary, digest, load_capture


ESCAPE_BONUS = 3.
SURVIVE_PAST = 16


def failed_prefixes(episodes, worker):
    prefixes, failures = [], []
    env = ReplayManagerEnv(episodes, worker)
    for index, episode in enumerate(episodes):
        for delay in (0, 1):
            env.reset(seed=0, options={'episode': index, 'delay': delay})
            states = {}
            while True:
                states[env.frame] = (env.position.copy(), env.belief, env.buffer, env.previous_capacity)
                _, _, done, _, info = env.step(0)
                if done:
                    break
            stop = env.frame
            if info['outcome'] not in ('collision', 'no_verified_route'):
                continue
            target = stop + SURVIVE_PAST
            selected = [frame for frame in (stop - 120, stop - 96, stop - 72, stop - 60)
                        if frame in states and target <= episode.end]
            failures.append({'source': episode.source, 'episode': episode.number, 'delay': delay,
                             'stop': stop, 'capture_end': episode.end, 'prefixes': selected})
            for frame in selected:
                prefixes.append((index, delay, frame, target, states[frame]))
    return prefixes, failures


class EscapeReplayEnv(ReplayManagerEnv):
    """Keep waypoint slots and reward verified escape from a matched rule failure."""

    def __init__(self, episodes, worker, prefixes=(), verify=False, forecast_on_reset=False):
        super().__init__(episodes, worker, verify=verify, attack_reward=True)
        self.prefixes = prefixes
        self.connectivity_until = None
        self.forecast_on_reset = forecast_on_reset

    def _prepare(self):
        if self.forecast_on_reset and self.frame == self.start:
            sample = self.episode.samples[self.frame]
            scene = replace(sample.scene, player=tuple(self.position))
            _, event, _ = forecast_horizon(scene)
            self.connectivity_until = self.frame + event['end'] if event else None
        obs = super()._prepare()
        in_contraction = self.connectivity_until is not None and self.frame <= self.connectivity_until
        if (self.danger or in_contraction) and self.choices:
            sample = self.episode.samples[self.frame]
            rebased = rebase(sample.observation, self.position)
            self.choices, edges, elapsed = self.worker.options(self.scene, rebased, self.belief,
                                                              waypoint_coverage=True)
            self.times[-1] += elapsed
            self.edges[-1] += edges
            obs = self._observation(rebased, sample)
        return obs

    def reset(self, *, seed=None, options=None):
        if options is not None or not self.prefixes:
            self.escape_target = None
            self.diverged = False
            self.connectivity_until = None
            return super().reset(seed=seed, options=options)
        # Initialize Gym's seed before sampling a prefix.
        if seed is not None:
            self._np_random, self._np_random_seed = self._seed(seed)
        index, delay, frame, target, state, self.connectivity_until = self.prefixes[
            int(self.np_random.integers(len(self.prefixes)))]
        self.escape_target = target
        self.diverged = False
        self.episode = self.episodes[index]
        self.frame = self.start = frame
        self.limit = target
        self.fixed_delay = delay
        self.position, self.belief, self.buffer, self.previous_capacity = state
        self.position = self.position.copy()
        self.times, self.edges, self.history = [], [], []
        self.max_omitted = 0
        self.min_gap = float('inf')
        self.padded_contact = self.last_padded_contact = None
        self.distinct_moves = 0
        return self._prepare(), {}

    @staticmethod
    def _seed(seed):
        from gymnasium.utils.seeding import np_random
        return np_random(seed)

    def step(self, action):
        if self.choices:
            self.diverged |= self.choices[int(action)].route.actions[0] != self.choices[0].route.actions[0]
        obs, reward, done, truncated, info = super().step(action)
        earned = (done and self.escape_target is not None and self.frame >= self.escape_target
                  and self.diverged and info['outcome'] in ('window_end', 'trace_end')
                  and self.padded_contact is None)
        info['escape_bonus'] = ESCAPE_BONUS if earned else 0.
        return obs, reward + info['escape_bonus'], done, truncated, info


class Stats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episodes = 0
        self.escapes = 0

    def _on_step(self):
        for info in self.locals['infos']:
            self.episodes += int(info['outcome'] is not None)
            self.escapes += int(info['escape_bonus'] > 0)
        return True

    def _on_rollout_end(self):
        print(f'{self.num_timesteps}: {self.escapes}/{self.episodes} training escapes', flush=True)


def evaluate_escape(episodes, worker, policy, cases):
    env = EscapeReplayEnv(episodes, worker, verify=True, forecast_on_reset=True)
    rows = []
    for case in cases:
        obs, _ = env.reset(seed=0, options=case)
        while True:
            obs, _, done, _, info = env.step(policy(obs))
            if done:
                break
        rows.append({'source': env.episode.source, 'episode': env.episode.number, 'delay': case['delay'], **info})
    return {'completed': sum(row['outcome'] == 'trace_end' for row in rows), 'total': len(rows),
            'mean_frames': float(np.mean([row['frames'] for row in rows])),
            'attack_position': attack_position_summary(episodes, rows), 'episodes': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manager', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    args = parser.parse_args()
    if args.steps < 128 or args.steps % 128:
        parser.error('--steps must be a positive multiple of 128')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_hash, base_hash = digest(args.manager), digest(BASE)
    train = [ep for path in TRAIN for ep in load_capture(path, base_hash)]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    prefixes, failures = failed_prefixes(train, worker)
    print(f'{len(failures)} failed rule branches; {len(prefixes)} eligible prefixes', flush=True)
    connectivity = []
    training_prefixes = []
    for index, delay, frame, target, state in prefixes:
        scene = replace(train[index].samples[frame].scene, player=tuple(state[0]))
        horizon, event, areas = forecast_horizon(scene)
        connectivity.append({'episode': index, 'delay': delay, 'frame': frame, 'target': target,
                             'event': event, 'forecast_horizon': horizon,
                             'samples': areas})
        training_prefixes.append((index, delay, frame, target, state, frame + event['end'] if event else None))
    report = {'frozen_sha256': base_hash, 'manager_source_sha256': source_hash,
              'training_sources': {str(path): digest(path) for path in TRAIN}, 'test_source_sha256': digest(TEST),
              'failures': failures, 'prefix_count': len(prefixes), 'connectivity': connectivity,
              'steps': args.steps, 'escape_bonus': ESCAPE_BONUS, 'evaluations': {}}

    def save():
        path = args.output / 'report.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        tmp.replace(path)

    save()
    if not prefixes:
        return
    env = EscapeReplayEnv(train, worker, training_prefixes)
    model = PPO.load(args.manager, env=env, device='cpu', seed=7)
    before = PPO.load(args.manager, device='cpu')
    stats = Stats()
    started = time.perf_counter()
    model.learn(args.steps, callback=stats)
    report.update(training_seconds=time.perf_counter() - started, training_episodes=stats.episodes,
                  training_escapes=stats.escapes)
    model.save(args.output / 'manager.zip')
    save()
    test = load_capture(TEST, base_hash)
    cases = [{'episode': i, 'delay': delay} for i in range(len(test)) for delay in (0, 1)]
    windows = [{'episode': 2, 'frame': frame, 'delay': delay} for frame in DIAGNOSTICS for delay in (0, 1)]
    for name, policy in [('before', before), ('trained', model)]:
        predict = lambda obs: policy.predict(obs, deterministic=True)[0].item()
        report['evaluations'][name] = {'full': evaluate_escape(test, worker, predict, cases),
                                       'known': evaluate_escape(test, worker, predict, windows)}
        print(name, [(key, result['completed'], result['mean_frames'])
                     for key, result in report['evaluations'][name].items()], flush=True)
        save()
    if digest(args.manager) != source_hash or digest(BASE) != base_hash:
        raise AssertionError('Source policy changed')


if __name__ == '__main__':
    main()
