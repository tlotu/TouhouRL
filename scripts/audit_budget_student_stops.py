"""Audit all first directions at the final held-out student decisions."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import HOLD, ReplayWorker, hold_outcomes, rebase
from scripts.diagnose_replay_mixed_windows import predict
from scripts.evaluate_replay_budget import MANAGERS, small_search
from scripts.train_replay_hrl import BASE, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    report = json.loads(args.report.read_text())
    if any(digest(Path(p)) != h for p, h in report['sources'].items()):
        raise ValueError('Source changed')
    torch.set_num_threads(1)
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(MANAGERS['old'], device='cpu')
    cases = []
    for fold in report['folds']:
        episodes = load_capture(Path(fold['held']), report['sources'][str(BASE)])
        if 'variants' in fold:
            runs = [(name, run) for name, group in fold['variants'].items() for run in group['rollouts']]
        else:
            runs = [(run['controller'], run) for run in fold['rollouts'] if run['controller'] == 'student']
        for name, run in runs:
            if run['outcome'] != 'no_verified_route':
                continue
            env = CoverageEnv(episodes, worker, verify=True)
            index = next(i for i, ep in enumerate(episodes) if ep.number == run['episode'])
            obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
            while env.frame < run['start']:
                obs, _, done, _, _ = env.step(predict(old, obs))
                if done:
                    raise AssertionError('Anchor unreachable')
            belief, position = env.belief, env.position
            states = []
            for i, row in enumerate(run['decisions']):
                sample = env.episode.samples[row['frame']]
                scene = replace(sample.scene, player=tuple(position))
                obs = rebase(sample.observation, position)
                if i >= len(run['decisions']) - 5:
                    allowed = feasible_actions(obs, HOLD)
                    short = [a for a in allowed if small_search(scene, belief, [a], 2)[0]]
                    wide, _, _ = worker.options(scene, obs, belief, by_direction=True)
                    states.append(dict(frame=row['frame'], shortlist=row['shortlist'],
                                       short_verified=short, full_verified=sorted(wide)))
                if i < len(run['history']):
                    h = run['history'][i]
                    if h['frame'] != row['frame'] or not np.array_equal(h['start'], position):
                        raise AssertionError('Student prefix changed')
                    outcomes = hold_outcomes(scene, belief, h['request'])
                    position = h['end']
                    belief = frozenset(outcomes[tuple(np.round(position, 6))])
            if set(states[-1]['short_verified']) & set(states[-1]['shortlist']):
                raise AssertionError('Short search disagrees with terminal result')
            cases.append(dict(held=fold['held'], controller=name, delay=run['delay'], states=states))
    with args.output.open('x') as stream:
        json.dump(dict(report_sha256=digest(args.report), cases=cases), stream, indent=2)


if __name__ == '__main__':
    main()

