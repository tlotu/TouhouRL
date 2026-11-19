"""Audit frozen-scene lookahead against known closed-loop action contrasts."""
import argparse
import json
from pathlib import Path
import time

import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import HOLD, ReplayWorker, rebase
from scripts.audit_replay_alternatives import seek
from scripts.audit_training_escape import physical_trace
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, args.old, args.anchors]}
    anchors = json.loads(args.anchors.read_text(encoding='utf-8'))
    if any(anchors['sources'].get(str(p)) != sources[str(p)] for p in [BASE, *TRAIN, args.old]):
        raise ValueError('Anchor sources differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(args.old, device='cpu')
    cases = []
    for source, origin, delay, target, actions in [
        (str(TRAIN[0]), 340, 0, 396, (8, 7)),
        (str(TRAIN[0]), 340, 1, 396, (8, 7)),
        (str(TRAIN[1]), 344, 0, 384, (7, 0)),
        (str(TRAIN[1]), 344, 0, 388, (3, 7)),
    ]:
        index = next(i for i, ep in enumerate(episodes) if ep.source == source and ep.number == 2)
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < origin:
            obs, _, done, _, _ = env.step(int(old.predict(obs, deterministic=True)[0]))
            if done:
                raise AssertionError('Stopped before anchor')
        env, obs = restore(env, duration=None)
        env.fixed_delay = delay
        while env.frame < target:
            obs, _, done, _, _ = env.step(int(old.predict(obs, deterministic=True)[0]))
            if done:
                raise AssertionError('Stopped before target')
        anchor = next(a for a in anchors['anchors'] if a['source'] == source and a['frame'] == origin)
        expected = next(r for r in anchor['policies']['old']['episodes'] if r['delay'] == delay)
        if physical_trace(env.history) != physical_trace([h for h in expected['history'] if h['frame'] < target]):
            raise AssertionError('Trajectory did not reproduce')
        allowed = feasible_actions(rebase(env.episode.samples[target].observation, env.position), HOLD)
        case = {'source': source, 'origin': origin, 'delay': delay, 'target': target,
                'position': env.position.tolist(), 'forecasts': []}
        for action in actions:
            for horizon in (16, 32, 60, 96):
                for model_delay in (0, 1):
                    started = time.perf_counter()
                    route, choice = seek(env.scene, env.belief, allowed, action, horizon, model_delay, 36)
                    row = {'action': action, 'horizon': horizon, 'model_delay': model_delay,
                           'found': route is not None,
                           'leaf_ttc': route.ttc if route else None,
                           'prefix_clearance': choice.gap if choice else None,
                           'actions': list(route.actions) if route else None,
                           'ms': 1000 * (time.perf_counter() - started)}
                    case['forecasts'].append(row)
                print(source, delay, target, action, horizon,
                      [(r['found'], r['leaf_ttc']) for r in case['forecasts'][-2:]], flush=True)
        cases.append(case)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'width': 36, 'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
