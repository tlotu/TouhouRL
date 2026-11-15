"""Train and evaluate the upper policy on separate Young Demon Lord capture logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent.forward_model import scene_from_observation
from agent.laser_policy import LASER_OBSERVATION_CONTRACT
from agent.ppo_backend import PPOBackend
from agent.replay_exploration import ReplayExplorationPolicy
from agent.replay_hrl import EPISODE_WINDOW, HOLD, MAX_EDGES, PADDING, SCALE, SHORT_HORIZON, WIDTH
from agent.replay_hrl import ReplayEpisode, ReplayManagerEnv, ReplaySample, ReplayWorker
from agent.replay_hrl import RISK_DELAY, RISK_THRESHOLD, SURVIVAL_HORIZON
from scripts.audit_buffered_intervention import frame_start, transitions
from scripts.replan_buffered_failure_windows import calibrate
from scripts.replay_buffered_closed_loop import digest


BASE = Path('runs/ppo-laser-young-smoke/checkpoints/update-000004/model.zip')
PREFIX = 'laser-planner-buffered-'
TRAIN = [Path('runs', PREFIX + name + '.jsonl') for name in ('leaf-shadow', 'reuse-shadow')]
TEST = Path('runs', PREFIX + 'input-profile.jsonl')
DIAGNOSTICS = (364, 380, 384, 396, 412, 416)


def load_capture(path, checkpoint_hash):
    episodes = []
    for header, rows, _ in transitions(path, allow_earlier=True):
        if (header.get('model_sha256') != checkpoint_hash or header.get('hold_frames') != HOLD
                or header.get('laser_filter', {}).get('observation_contract') != LASER_OBSERVATION_CONTRACT):
            raise ValueError(f'Incompatible capture: {path}')
        if any(a['info']['elapsed_frames'] != frame_start(b) or a['next_observation'] != b['observation']
               for a, b in zip(rows, rows[1:])):
            raise ValueError(f'Non-contiguous observations: {path}')
        observations = [{key: np.asarray(value, dtype=np.float32) for key, value in row['observation'].items()}
                        for row in rows]
        scenes = [scene_from_observation(obs)[0] for obs in observations]
        priors, _ = calibrate(rows, scenes)
        samples = {frame_start(row): ReplaySample(obs, scene, previous['info']['dropped_bullets'] if previous else 0)
                   for row, obs, scene, previous in zip(rows, observations, scenes, [None, *rows[:-1]])}
        end = rows[-1]['info']['elapsed_frames']
        if sorted(samples) != list(range(0, end, 2)):
            raise ValueError(f'Expected observations every two frames: {path}')
        episodes.append(ReplayEpisode(str(path), rows[0]['episode'] + 1, samples, priors, end))
    return episodes


def evaluate(episodes, worker, policy, cases):
    env = ReplayManagerEnv(episodes, worker, verify=True)
    rows = []
    for case in cases:
        obs, _ = env.reset(seed=0, options=case)
        reward_sum = 0.
        while True:
            obs, reward, done, truncated, info = env.step(policy(obs))
            reward_sum += reward
            if done or truncated:
                break
        rows.append({'source': env.episode.source, 'episode': env.episode.number, 'delay': case['delay'],
                     'trace_end_frame': env.episode.end, 'return': reward_sum, **info})
    times = [ms for row in rows for ms in row['planning_ms']]
    return {'outcomes': dict(Counter(row['outcome'] for row in rows)),
            'mean_frames': float(np.mean([row['frames'] for row in rows])),
            'below_padding': sum(row['first_below_padding_frame'] is not None for row in rows),
            'distinct_move_decisions': sum(row['distinct_move_decisions'] for row in rows),
            'planning_ms_p95': float(np.percentile(times, 95)),
            'decisions_over_16_6ms': sum(ms > 16.6 for ms in times),
            'episodes': rows}


def attack_position_summary(episodes, rows):
    """Measure below-boss horizontal alignment at decisions, without modeling damage."""
    captures = {(ep.source, ep.number): ep for ep in episodes}
    gaps = []
    inside = 0
    for row in rows:
        episode = captures[row['source'], row['episode']]
        for move in row['history']:
            sample = episode.samples[move['frame']]
            enemies = sample.observation['enemies']
            boss = enemies[(enemies[:, 2] > 0) & (enemies[:, 3:6] > 0).all(axis=1)]
            if len(boss):
                x, y = np.asarray(sample.scene.player) + boss[0, :2] * (2 * SCALE)
                dx = abs(float(x) - move['start'][0])
                gaps.append(dx)
                inside += int(dx <= 24 and move['start'][1] >= y)
    return {'boss_observed_decisions': len(gaps), 'below_boss_within_24px_decisions': inside,
            'mean_horizontal_gap_px': float(np.mean(gaps)) if gaps else None}


def paired_move_summary(report):
    results = {}
    for group in ('full_trace', 'known_windows'):
        trained = report['evaluations']['trained'][group]['episodes']
        rule = report['evaluations']['rule'][group]['episodes']
        pairs = list(zip(trained, rule, strict=True))
        comparisons = [(a, b) for first, second in pairs
                       for a, b in zip(first['history'], second['history'])]
        results[group] = {'common_decisions': len(comparisons),
                          'different_requests': sum(a['request'] != b['request'] for a, b in comparisons),
                          'different_outcomes': sum(a['outcome'] != b['outcome'] for a, b in pairs)}
    return results


class TrainingStats(BaseCallback):
    def __init__(self):
        super().__init__()
        self.outcomes = Counter()
        self.rollouts = []
        self.penalties = self.bonuses = 0
        self.danger_steps = self.attack_steps = 0

    def _on_step(self):
        for info in self.locals['infos']:
            self.penalties += int(info['continuation_penalty'] < 0.)
            self.bonuses += int(info['survival_bonus'] > 0.)
            self.danger_steps += int(info['danger'])
            self.attack_steps += int(info['attack_reward'] > 0.)
            if info['outcome'] is not None:
                self.outcomes[info['outcome']] += 1
        return True

    def _on_rollout_end(self):
        self.rollouts.append({'step': self.num_timesteps, 'outcomes': dict(self.outcomes),
                              'penalty_steps': self.penalties, 'bonus_steps': self.bonuses,
                              'danger_steps': self.danger_steps, 'attack_steps': self.attack_steps})
        print(f'Training {self.num_timesteps}: {dict(self.outcomes)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=7)
    reward = parser.add_mutually_exclusive_group()
    reward.add_argument('--long-reward', action='store_true')
    reward.add_argument('--conditional-reward', action='store_true')
    parser.add_argument('--attack-reward', action='store_true')
    parser.add_argument('--exploration', type=float, default=0.)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 128:
        parser.error('--steps must be a positive multiple of 128')
    if not 0. <= args.exploration < 1.:
        parser.error('--exploration must be in [0, 1)')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    frozen_hash = digest(BASE)
    train = [ep for path in TRAIN for ep in load_capture(path, frozen_hash)]
    test = load_capture(TEST, frozen_hash)
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    backend.model.policy.requires_grad_(False).eval()
    worker = ReplayWorker(backend.action_logits)
    env = ReplayManagerEnv(train, worker, long_reward=args.long_reward,
                           conditional_reward=args.conditional_reward, attack_reward=args.attack_reward)
    if any(not episode.start_frames() for episode in train):
        raise ValueError('Each training episode requires calibrated start frames')
    print(f'{len(train)} training episodes, {len(env.starts)} calibrated starts, {len(test)} held-out episodes',
          flush=True)
    policy = ReplayExplorationPolicy if args.exploration else 'MlpPolicy'
    policy_kwargs = {'net_arch': [64, 64]}
    if args.exploration:
        policy_kwargs['exploration'] = args.exploration
    model = PPO(policy, env, seed=args.seed, n_steps=128, batch_size=64, n_epochs=8,
                gamma=.99, learning_rate=.0005, policy_kwargs=policy_kwargs, device='cpu')
    holdout = [{'episode': i, 'delay': d} for i in range(len(test)) for d in (0, 1)]
    cases = [{'episode': 2, 'frame': f, 'delay': d} for f in DIAGNOSTICS for d in (0, 1)]
    report = {'contract': 'young-demon-lord-replay-ppo-v1', 'seed': args.seed, 'steps': args.steps,
              'frozen_sha256': frozen_hash, 'padding_px': PADDING, 'hold_frames': HOLD,
              'lookahead_frames': SHORT_HORIZON, 'window_frames': EPISODE_WINDOW,
              'beam_width': WIDTH, 'max_edges_per_search': MAX_EDGES,
              'exploration': args.exploration,
              'conditional_reward': args.conditional_reward,
              'attack_reward': env.attack_reward,
              'long_reward': {'continuation_delay': RISK_DELAY, 'ttc_threshold': RISK_THRESHOLD,
                              'survival_horizon': SURVIVAL_HORIZON}
              if args.long_reward or args.conditional_reward else None,
              'sources': {str(path): digest(path) for path in [*TRAIN, TEST]},
              'training': [{'source': ep.source, 'episode': ep.number, 'calibrated_starts': len(ep.start_frames())}
                           for ep in train], 'held_out_source': str(TEST), 'evaluations': {}}

    def save():
        target = args.output / 'report.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(target)

    def predict(obs):
        return model.predict(obs, deterministic=True)[0].item()

    def measure(label, policy):
        report['evaluations'][label] = {key: evaluate(test, worker, policy, group)
                                         for key, group in [('full_trace', holdout), ('known_windows', cases)]}
        for result in report['evaluations'][label].values():
            result['attack_position'] = attack_position_summary(test, result['episodes'])
        result = report['evaluations'][label]['full_trace']
        print(f"{label}: {result['outcomes']}; mean {result['mean_frames']:.1f} frames", flush=True)
        save()

    # Both snapshots of PPO are evaluated after training so the same code path
    # and held-out logs are used for all controllers.
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    stats = TrainingStats()
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=stats)
    report.update(training_seconds=time.perf_counter() - started, training_rollouts=stats.rollouts)
    model.save(args.output / 'manager.zip')
    save()
    measure('trained', predict)
    model.policy.load_state_dict(before)
    measure('untrained', predict)
    measure('rule', lambda _: 0)
    report['trained_vs_rule'] = paired_move_summary(report)
    if digest(BASE) != frozen_hash or any(p.requires_grad for p in backend.model.policy.parameters()):
        raise AssertionError('Frozen actor changed')
    report['worker_unchanged'] = True
    save()


if __name__ == '__main__':
    main()
