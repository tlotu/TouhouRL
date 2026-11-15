"""Supervise a PPO manager with paired first-move replay outcomes."""
import argparse
from copy import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from agent.replay_hrl import HOLD, OPTIONS, PADDING, SHORT_HORIZON, ReplayManagerEnv, ReplayWorker, rebase
from agent.short_planner import exit_hint
from scripts.train_replay_hrl import BASE, DIAGNOSTICS, TEST, TRAIN, PPO, PPOBackend
from scripts.train_replay_hrl import attack_position_summary, digest, evaluate, load_capture


def event_horizon(scene):
    padded = replace(scene, player_radius=scene.player_radius + PADDING)
    event = exit_hint(padded, nearest_closure=True)
    if event is None:
        return None, 72
    return event.closing_frames, max(SHORT_HORIZON, HOLD * math.ceil((event.closing_frames + 5) / HOLD))


def expanded_options(env):
    sample = env.episode.samples[env.frame]
    obs = rebase(sample.observation, env.position)
    options, _, _ = env.worker.options(env.scene, obs, env.belief, by_direction=True)
    original = env.choices
    env.choices = tuple(options.get(i) for i in range(OPTIONS))
    encoded = env._observation(obs, sample)
    env.choices = original
    return options, encoded


def fork(env, buffer, delay, frames):
    branch = copy(env)
    branch.buffer = buffer
    branch.fixed_delay = delay
    branch.limit = min(env.episode.end, env.frame + frames)
    branch.times = []
    branch.edges = []
    branch.history = []
    return branch


def outcome(env, choice, frames):
    results = []
    for buffer in sorted(env.belief, key=repr):
        for delay in (0, 1):
            branch = fork(env, buffer, delay, frames)
            branch.choices = (choice,)
            origin = branch.frame
            branch.padded_contact = None
            score = 0.
            discount = 1.
            while True:
                _, reward, done, _, info = branch.step(0)
                score += discount * reward
                discount *= .99
                if done:
                    break
            results.append({'delay': delay, 'frames': branch.frame - origin,
                            'outcome': info['outcome'], 'below_padding': branch.padded_contact is not None,
                            'return': score})
    return results


def collect(episodes, worker, per_episode):
    data = []
    for index, episode in enumerate(episodes):
        env = ReplayManagerEnv(episodes, worker, verify=True)
    obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 1})
        last = -100
        selected = 0
        while selected < per_episode:
            if (env.danger and env.choices and env.frame >= 180 and env.frame - last >= 24
                    and episode.end - env.frame >= SHORT_HORIZON):
                origin = env.frame
                closing, horizon = event_horizon(env.scene)
                if origin + horizon <= episode.end:
                    options, encoded = expanded_options(env)
                    baseline_direction = env.choices[0].route.actions[0]
                    trials = {direction: outcome(env, choice, horizon) for direction, choice in options.items()}
                    eligible = [direction for direction, rows in trials.items()
                                if all(row['frames'] == horizon and not row['below_padding'] for row in rows)]
                    alternatives = eligible if baseline_direction not in eligible else [baseline_direction]
                    data.append({'source': episode.source, 'episode': episode.number, 'frame': origin,
                                 'closing_frame': origin + closing if closing is not None else None,
                                 'horizon': horizon, 'observation': encoded.tolist(),
                                 'baseline_direction': baseline_direction, 'trials': trials,
                                 'labels': alternatives, 'eligible': eligible})
                    print(f'{Path(episode.source).stem} ep {episode.number} frame {origin}: '
                          f'{len(options)} directions, {len(eligible)} survived, baseline {baseline_direction}', flush=True)
                    last = origin
                    selected += 1
            obs, _, done, _, _ = env.step(0)
            if done:
                break
    return data


class DangerEvalEnv(ReplayManagerEnv):
    """Expand the candidate set at danger triggers and mask unavailable directions."""

    def _prepare(self):
        obs = super()._prepare()
        self.direction_options = {}
        if self.danger and self.choices:
            self.direction_options, obs = expanded_options(self)
        return obs

    def step(self, action):
        if self.direction_options:
            self.choices = (self.direction_options.get(action, self.choices[0]),)
        return super().step(0)


def evaluate_danger(episodes, worker, policy, cases):
    env = DangerEvalEnv(episodes, worker, verify=True)
    rows = []
    for case in cases:
        obs, _ = env.reset(seed=0, options=case)
        interventions = 0
        while True:
            action = None
            if env.direction_options:
                action = policy(obs, env.direction_options)
                interventions += int(action is not None and action != env.choices[0].route.actions[0])
            obs, _, done, _, info = env.step(action)
            if done:
                break
        rows.append({'source': env.episode.source, 'episode': env.episode.number,
                     'delay': case['delay'], 'interventions': interventions, **info})
    return {'completed': sum(r['outcome'] == 'trace_end' for r in rows),
            'total': len(rows), 'mean_frames': float(np.mean([r['frames'] for r in rows])),
            'attack_position': attack_position_summary(episodes, rows), 'episodes': rows}


def supervise(model, data):
    examples = [row for row in data if row['labels']]
    obs = torch.tensor([row['observation'] for row in examples], dtype=torch.float32)
    legal = torch.tensor([[a in row['trials'] for a in range(OPTIONS)] for row in examples])
    targets = torch.tensor([[float(a in row['labels']) / len(row['labels']) for a in range(OPTIONS)]
                            for row in examples])
    values = torch.tensor([min(result['return'] for a in row['labels'] for result in row['trials'][a])
                           for row in examples], dtype=torch.float32)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0005)
    losses = []
    for _ in range(80):
        features = policy.extract_features(obs)
        actor, critic = policy.mlp_extractor(features)
        logits = policy.action_net(actor).masked_fill(~legal, -1e9)
        bc = -(targets * logits.log_softmax(-1)).sum(-1).mean()
        value = (policy.value_net(critic).squeeze(-1) - values).square().mean()
        loss = bc + .25 * value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append({'bc': float(bc.detach()), 'value_mse': float(value.detach())})
    return losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manager', type=Path, required=True)
    parser.add_argument('--per-episode', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = digest(BASE)
    manager_hash = digest(args.manager)
    train = [episode for source in TRAIN for episode in load_capture(source, checkpoint)]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits)
    data = collect(train, worker, args.per_episode)
    model = PPO.load(args.manager, device='cpu')
    before = PPO.load(args.manager, device='cpu')
    (args.output / 'demonstrations.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    losses = supervise(model, data)
    model.save(args.output / 'manager.zip')
    test = load_capture(TEST, checkpoint)
    full = [{'episode': i, 'delay': d} for i in range(len(test)) for d in (0, 1)]
    known = [{'episode': 2, 'frame': f, 'delay': d} for f in DIAGNOSTICS for d in (0, 1)]

    def chooser(policy):
        @torch.no_grad()
        def choose(obs, options):
            actor = policy.mlp_extractor.forward_actor(policy.extract_features(torch.from_numpy(obs[None, :])))
            logits = policy.action_net(actor)[0]
            return max(options, key=lambda action: logits[action].item())
        return choose

    report = {'training_sources': {str(path): digest(path) for path in TRAIN},
              'test_source_sha256': digest(TEST), 'base_sha256': checkpoint,
              'manager_source_sha256': manager_hash, 'training_states': len(data),
              'expert_states': sum(bool(row['labels']) for row in data),
              'improvement_states': sum(bool(row['labels']) and row['baseline_direction'] not in row['eligible']
                                        for row in data),
              'losses': losses, 'evaluations': {}}
    for name, choose in [('expanded_before', chooser(before.policy)), ('supervised', chooser(model.policy))]:
        report['evaluations'][name] = {key: evaluate_danger(test, worker, choose, cases)
                                       for key, cases in [('full', full), ('known', known)]}
        print(name, [(key, result['completed'], result['mean_frames'])
                     for key, result in report['evaluations'][name].items()], flush=True)
    if digest(BASE) != checkpoint or digest(args.manager) != manager_hash:
        raise AssertionError('Source checkpoint changed')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
