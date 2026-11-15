"""Measure speculative planning under a paced replay clock."""
from collections import Counter
import json
from pathlib import Path
import argparse
import time

import numpy as np
import torch

from agent.replay_async import AsyncReplayPlanner, fallback_hold
from agent.replay_hrl import HOLD, ReplayManagerEnv, ReplayWorker, rebase
from scripts.train_replay_hrl import BASE, TEST, PPO, PPOBackend, digest, load_capture


class PacedReplayEnv(ReplayManagerEnv):
    def __init__(self, episodes, worker, planner):
        super().__init__(episodes, worker, verify=True)
        self.planner = planner
        self.counts = Counter()
        self.consume_ms = []
        self.boundary_ms = []
        self.submit_ms = []
        self.deadline = None
        self.previous = 0

    def _plan_options(self, obs):
        if self.deadline is not None:
            remaining = self.deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        started = time.perf_counter()
        candidate, status = self.planner.consume(self.frame, self.scene, obs, self.belief)
        consume = 1000 * (time.perf_counter() - started)
        self.consume_ms.append(consume)
        self.counts[status] += 1
        if candidate is None:
            candidate = fallback_hold(self.scene, obs, self.belief, self.previous)
            self.counts['fallback_hold' if candidate else 'fallback_unavailable'] += 1
        elapsed = 1000 * (time.perf_counter() - started)
        self.boundary_ms.append(elapsed)
        return ((candidate,) if candidate is not None else ()), 0, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=384)
    args = parser.parse_args()
    torch.set_num_threads(1)
    episodes = load_capture(TEST, digest(BASE))
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits)
    planner = AsyncReplayPlanner(worker)
    env = PacedReplayEnv(episodes, worker, planner)
    window = HOLD / 60
    completed = []
    try:
        for delay in (0, 1):
            planner.reset()
            env.deadline = None
            env.previous = 0
            env.reset(seed=0, options={'episode': 2, 'frames': args.frames, 'delay': delay})
            first_boundary = len(env.boundary_ms) - 1
            first_submit = len(env.submit_ms)
            done = False
            while env.choices:
                chosen = env.choices[0]
                started = time.perf_counter()
                sample = env.episode.samples[env.frame]
                submitted = planner.submit(env.frame, env.scene, rebase(sample.observation, env.position),
                                           env.belief, chosen.route.actions[0])
                env.submit_ms.append(1000 * (time.perf_counter() - started))
                env.counts['submitted' if submitted else 'not_submitted'] += 1
                env.previous = chosen.route.actions[0]
                env.deadline = started + window
                _, _, done, _, info = env.step(0)
                if done:
                    break
            if not done:
                _, _, _, _, info = env.step(0)
            missing = len(env.boundary_ms) - first_boundary - (len(env.submit_ms) - first_submit)
            env.submit_ms.extend([0.] * missing)
            completed.append({'delay': delay, 'frame': env.frame, 'outcome': info['outcome'],
                              'below_padding': info['first_below_padding_frame'] is not None})
    finally:
        planner.close()
    main_ms = np.asarray(env.boundary_ms) + np.asarray(env.submit_ms)
    report = {'budget_ms': 1000 * window, 'states': dict(env.counts), 'branches': completed,
              'main_thread_consume_p95_ms': float(np.percentile(env.consume_ms, 95)),
              'main_thread_boundary_p95_ms': float(np.percentile(env.boundary_ms, 95)),
              'main_thread_boundary_with_submit_p95_ms': float(np.percentile(main_ms, 95)),
              'main_thread_over_16_6ms': int(np.count_nonzero(main_ms > 16.6)),
              'scope': 'Paced replay; boundary includes four-frame heuristic fallback; no synchronous beam'}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(report)


if __name__ == '__main__':
    main()
