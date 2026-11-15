"""Train waypoint PPO with the existing worker and a buffered beam executor."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from agent.buffered_hrl import BufferedManagerEnv, BufferedWorker, buffered_tasks, validation_tasks
from agent.offline_hrl import GoalWorker, state_digest


def evaluate(tasks, worker, policy, expose_buffer=True):
    rows = []
    for delay in (0, 1):
        env = BufferedManagerEnv(tasks, worker, delay, verify=True, expose_buffer=expose_buffer)
        for index, task in enumerate(tasks):
            observation, _ = env.reset(options={'task': index})
            choices = []
            done = False
            total_reward = 0.
            while not done:
                option = int(policy(env, observation))
                choices.append(option)
                observation, reward, done, _, info = env.step(option)
                total_reward += reward
            rows.append({'task': task.name, 'delay': delay, 'options': choices, 'return': total_reward, **info})
    completed = [r['frame'] for r in rows if r['success']]
    return {'passed': len(completed), 'total': len(rows), 'mean_completion_frames': statistics.mean(completed)
            if completed else None, 'mean_return': statistics.mean(r['return'] for r in rows), 'episodes': rows}


def timed_rule(env, _):
    if not env.scene.lasers or env.task.crossed(env.scene.player):
        return 2
    x, y = env.scene.player
    wall = env.scene.lasers[0]
    lane = env.scene.bounds[2] + (6, 34)[env.task.opening]
    direct_crossing = abs(x - wall.ax) / env.scene.speed
    detour_crossing = direct_crossing + abs(y - lane) / env.scene.speed
    return 2 if wall.off <= detour_crossing else env.task.opening


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidationCurve(BaseCallback):
    """Evaluate at completed rollout updates with a separate fixed task set."""

    def __init__(self, interval, evaluate_manager, curve, save):
        super().__init__()
        self.interval, self.evaluate_manager, self.curve, self.save = interval, evaluate_manager, curve, save

    def _record(self):
        if self.curve and self.curve[-1]['step'] == self.num_timesteps:
            return
        result = self.evaluate_manager()
        result.pop('episodes')
        self.curve.append({'step': self.num_timesteps, **result})
        self.save()
        print(f"Validation {self.num_timesteps}: {result['passed']}/{result['total']} "
              f"(return {result['mean_return']:.3f})", flush=True)

    def _on_rollout_start(self):
        if self.num_timesteps % self.interval == 0:
            self._record()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self._record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/ppo-laser-young-smoke/checkpoints/update-000004/model.zip'))
    parser.add_argument('--worker', type=Path, default=Path('runs/hrl-frozen-smoke-001/worker.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--hide-buffer', action='store_true', help='Hide buffer state and delay from the manager only')
    parser.add_argument('--eval-every', type=int, default=4096, help='Validation interval in manager decisions')
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 128:
        parser.error('--steps must be a positive multiple of 128')
    if args.eval_every <= 0 or args.eval_every % 128:
        parser.error('--eval-every must be a positive multiple of 128')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source_hash, worker_archive_hash = digest(args.source), digest(args.worker)
    archive = torch.load(args.worker, map_location='cpu', weights_only=True)
    if archive['source_sha256'] != source_hash:
        raise ValueError('Worker adapter requires its original checkpoint')
    worker = GoalWorker(PPO.load(args.source, device='cpu').policy)
    worker.adapter.load_state_dict(archive['adapter'])
    worker_hash = state_digest(worker)
    if worker_hash != archive['worker_sha256']:
        raise ValueError('Worker state digest mismatch')
    executor = BufferedWorker(worker)
    held_out, validation = buffered_tasks(True), validation_tasks()
    training = BufferedManagerEnv(buffered_tasks(), executor, delay=None, expose_buffer=not args.hide_buffer)
    model = PPO('MlpPolicy', training, seed=args.seed, n_steps=128, batch_size=64, n_epochs=8,
                gamma=.98, learning_rate=.0005, policy_kwargs={'net_arch': [32, 32]}, device='cpu')

    def predict(_, obs):
        return model.predict(obs, deterministic=True)[0].item()

    args.output.mkdir(parents=True, exist_ok=False)
    report = {'contract': 'offline-buffered-waypoint-ppo-v1', 'seed': args.seed, 'requested_steps': args.steps,
              'manager_observes_buffer': not args.hide_buffer,
              'source_sha256': source_hash, 'worker_archive_sha256': worker_archive_hash,
              'worker_sha256': worker_hash, 'training_layouts': len(training.tasks), 'held_out_layouts': len(held_out),
              'validation_layouts': len(validation), 'eval_every': args.eval_every, 'validation_curve': [],
              'scope': 'Synthetic timed laser detours with modeled direction event queue and zero/one-frame start delay.'}
    target = args.output / 'report.json'

    def save():
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    report['rule'] = evaluate(held_out, executor, timed_rule)
    report['direct'] = evaluate(held_out, executor, lambda *_: 2)
    report['before'] = evaluate(held_out, executor, predict, expose_buffer=not args.hide_buffer)
    report['validation_rule'] = evaluate(validation, executor, timed_rule)
    save()
    print(f"Rule {report['rule']['passed']}/{report['rule']['total']}. Training {args.steps} manager decisions.", flush=True)
    started = time.perf_counter()
    callback = ValidationCurve(args.eval_every,
                               lambda: evaluate(validation, executor, predict, expose_buffer=not args.hide_buffer),
                               report['validation_curve'], save)
    model.learn(total_timesteps=args.steps, callback=callback)
    report['training_seconds'] = time.perf_counter() - started
    report['after'] = evaluate(held_out, executor, predict, expose_buffer=not args.hide_buffer)
    if state_digest(worker) != worker_hash or digest(args.source) != source_hash or digest(args.worker) != worker_archive_hash:
        raise AssertionError('Frozen worker or source changed during training')
    model.save(args.output / 'manager.zip')
    report.update(manager_steps=model.num_timesteps, worker_unchanged=True)
    save()
    for key in ('direct', 'rule', 'before', 'after'):
        result = report[key]
        print(f"{key}: {result['passed']}/{result['total']} with mean completion {result['mean_completion_frames']} frames")
    print(target.resolve())


if __name__ == '__main__':
    main()
