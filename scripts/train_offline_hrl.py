"""Fit a goal adapter, validate it, then train PPO with the worker frozen."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from stable_baselines3 import PPO
import torch
from torch.nn.functional import cross_entropy

from agent.offline_hrl import DetourManagerEnv, GoalWorker, detour_tasks, state_digest, worker_examples
from scripts.replay_planning import write_replay


def _accuracy(worker, data):
    with torch.no_grad():
        return float((worker.logits(*data[:3]).argmax(1) == data[3]).float().mean())


def _fit(worker, training, seed):
    optimizer = torch.optim.Adam(worker.adapter.parameters(), lr=.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(45):
        for index in torch.randperm(len(training[-1]), generator=generator).split(128):
            features, prior, mask, labels = (value[index] for value in training)
            loss = cross_entropy(worker.logits(features, prior, mask), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def _evaluate(tasks, worker, name, policy):
    env = DetourManagerEnv(tasks, worker, record=True)
    episodes, tracks = [], []
    for index, task in enumerate(tasks):
        observation, _ = env.reset(options={'task': index})
        done = False
        choices = []
        while not done:
            choice = int(policy(env, observation))
            choices.append(choice)
            observation, _, done, _, info = env.step(choice)
        episodes.append({'task': task.name, **info, 'options': choices})
        tracks.append({'label': f'{name} / {task.name}', 'scene': asdict(task.scene), 'goal': asdict(task.goal),
                       'frames': env.decisions, 'source': 'Offline HRL model rollout',
                       'outcome': 'Reached goal' if info['success'] else 'Missed goal'})
    return {'passed': sum(e['success'] for e in episodes), 'total': len(episodes), 'episodes': episodes}, tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/ppo-laser-young-smoke/checkpoints/update-000004/model.zip'))
    parser.add_argument('--output', type=Path, default=Path('runs/hrl-frozen-smoke'))
    parser.add_argument('--steps', type=int, default=8192)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 128:
        parser.error('--steps must be a positive multiple of 128')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    worker = GoalWorker(PPO.load(args.source, device='cpu').policy)
    prior_hash = state_digest(worker.base)
    training_tasks, held_out = detour_tasks(), detour_tasks(held_out=True)
    training = worker_examples(worker, training_tasks, 4096, args.seed)
    validation = worker_examples(worker, held_out, 1024, args.seed + 1)
    accuracy_before = _accuracy(worker, validation)
    _fit(worker, training, args.seed)
    worker.eval()
    accuracy_after = _accuracy(worker, validation)
    def guide(env, _):
        aligned = abs(env.scene.player[0] - env.task.goal.x) <= env.task.goal.radius
        return 2 if aligned else env.task.opening

    guided, guide_tracks = _evaluate(held_out, worker, 'Guided worker', guide)
    output = args.output
    suffix = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            output = args.output.with_name(f'{args.output.name}-{suffix:03d}')
    report = {
        'contract': 'offline-static-hrl-v1', 'seed': args.seed, 'source': str(args.source.resolve()),
        'source_sha256': source_hash, 'training_examples': len(training[-1]),
        'held_out_examples': len(validation[-1]), 'held_out_accuracy_before': accuracy_before,
        'held_out_accuracy_after': accuracy_after, 'guided_worker': guided, 'tracks': guide_tracks,
    }

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    print(f'Guided worker {guided["passed"]}/{guided["total"]}, held-out action accuracy {accuracy_after:.3f}', flush=True)
    save()
    if guided['passed'] != guided['total']:
        raise RuntimeError(f'Worker failed tracking gate. See {output / "report.json"}')
    worker.requires_grad_(False).eval()
    frozen_hash = state_digest(worker)
    torch.save({'source_sha256': source_hash, 'adapter': worker.adapter.state_dict(),
                'worker_sha256': frozen_hash}, output / 'worker.pt')
    env = DetourManagerEnv(training_tasks, worker)
    model = PPO('MlpPolicy', env, seed=args.seed, n_steps=128, batch_size=64, n_epochs=8,
                gamma=.98, learning_rate=.0005, policy_kwargs={'net_arch': [32, 32]}, device='cpu')

    def predict(_, observation):
        return model.predict(observation, deterministic=True)[0].item()

    before, baseline_tracks = _evaluate(held_out, worker, 'Untrained manager', predict)
    model.learn(total_timesteps=args.steps)
    after, trained_tracks = _evaluate(held_out, worker, 'Trained manager', predict)
    if (state_digest(worker) != frozen_hash or state_digest(worker.base) != prior_hash
            or hashlib.sha256(args.source.read_bytes()).hexdigest() != source_hash):
        raise AssertionError('Frozen worker or source changed during training')
    model.save(output / 'manager.zip')
    report.update(manager_steps=model.num_timesteps, manager_before=before, manager_after=after,
                  frozen_worker_sha256=frozen_hash, worker_unchanged=True,
                  tracks=trained_tracks + baseline_tracks + guide_tracks)
    save()
    write_replay(report, output / 'replay.html')
    print(f'Manager {before["passed"]}/{before["total"]} -> {after["passed"]}/{after["total"]}')
    print((output / 'report.json').resolve())


if __name__ == '__main__':
    main()
