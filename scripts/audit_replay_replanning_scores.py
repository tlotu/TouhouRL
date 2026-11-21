"""Rank first moves by receding-horizon rollout of currently visible hazards."""
import argparse
from copy import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent.replay_async import project
from agent.replay_hrl import HOLD, ReplayWorker, hold_outcomes, rebase
from agent.short_planner import VECTORS
from scripts.audit_training_escape import physical_trace
from scripts.diagnose_replay_mixed_windows import predict
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


def forecast(origin, action, worker, horizon=60, policy=None, *, sample=None, reuse_options=True):
    if sample is None:
        sample = origin.episode.samples[origin.frame]
    return _forecast(origin, action, worker, horizon, policy, sample, {} if reuse_options else None)


def forecast_candidates(origin, actions, worker, horizon, policy=None, *, can_win=None):
    """Share deterministic choices across candidates from one observation."""
    sample = origin.episode.samples[origin.frame]
    options = {}
    for action in actions:
        yield action, _forecast(origin, action, worker, horizon, policy, sample, options, can_win=can_win)


def _forecast(origin, action, worker, horizon, policy, sample, options, *, can_win=None):
    observation = sample.observation
    view = copy(origin)
    branches = []
    branch_count = 2 * len(origin.belief)
    horizon_cap = ((horizon + HOLD - 1) // HOLD) * HOLD
    finished_frames = 0
    for initial in sorted(origin.belief, key=lambda q: (q.requested, q.applied, q.pending)):
        for delay in (0, 1):
            scene, queue, belief = origin.scene, initial, origin.belief
            elapsed, first = 0, True
            requests = []
            while elapsed < horizon:
                # Within this forecast, elapsed time fixes every hazard's geometry.
                key = elapsed, tuple(scene.player), belief
                if options is not None and key in options:
                    choices = options[key]
                else:
                    choices, _, _ = worker.options(scene, rebase(observation, scene.player), belief,
                                                    waypoint_coverage=True)
                    if options is not None:
                        options[key] = choices
                if not choices:
                    break
                if first and action is not None:
                    selected = next((c for c in choices if c.route.actions[0] == action), None)
                    if selected is None:
                        break
                else:
                    slot = 0
                    if policy is not None:
                        view.scene, view.belief, view.choices = scene, belief, choices
                        view.position = np.asarray(scene.player)
                        features = view._observation(rebase(observation, scene.player), sample)
                        slot = predict(policy, features)
                    selected = choices[slot]
                first = False
                move = selected.route.actions[0]
                outcomes = hold_outcomes(scene, belief, move)
                motions, queue = queue.advance(move, HOLD, delay)
                position = np.asarray(scene.player) + scene.speed * VECTORS[list(motions)].sum(axis=0)
                belief = frozenset(outcomes[tuple(np.round(position, 6))])
                if queue not in belief:
                    raise AssertionError('Projected queue removed')
                scene = project(scene, tuple(position))
                elapsed += HOLD
                requests.append(move)
            branches.append({'delay': delay, 'frames': elapsed, 'requests': requests})
            finished_frames += elapsed
            remaining = branch_count - len(branches)
            if can_win is not None and remaining:
                ceiling = (finished_frames + remaining * horizon_cap) / branch_count
                if not can_win(action, ceiling):
                    return None
    return {'branches': branches, 'worst_frames': min(b['frames'] for b in branches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tail', choices=('geometry', 'ppo'), default='geometry')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    anchor_path = Path('runs/hrl-young-delay-anchors-001.json')
    contrast_path = Path('runs/hrl-young-route-contraction-001.json')
    paths = {'old': Path('runs/hrl-young-replay-smoke-001/manager.zip'),
             'paired': Path('runs/hrl-young-delay-paired-001/manager.zip')}
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, anchor_path, contrast_path, *paths.values()]}
    anchors = json.loads(anchor_path.read_text(encoding='utf-8'))
    contrasts = json.loads(contrast_path.read_text(encoding='utf-8'))
    for report in (anchors, contrasts):
        if any(sources.get(p) != h for p, h in report['sources'].items() if p in sources):
            raise ValueError('Sources differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    actual = ReplayWorker(backend.action_logits, prune_failed=True)
    geometric = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True)
    policies = {name: PPO.load(p, device='cpu') for name, p in paths.items()}
    cases = []
    for anchor in anchors['anchors']:
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        env = CoverageEnv(episodes, actual, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Early termination')
        for contrast in contrasts['cases']:
            if contrast['source'] != anchor['source']:
                continue
            # Fixed 50% contraction rule, evaluated at previously tested decision frames.
            scan = contrast['scan'][contrast['failed_policy']]
            triggered = {b['frame'] for a, b in zip(scan, scan[1:])
                         if b['baseline_request'] is not None and a['layers'][-1]['sequences'] > 0
                         and b['layers'][-1]['sequences'] <= .5 * a['layers'][-1]['sequences']}
            targets = sorted(triggered & {r['frame'] for r in contrast['trials']})
            branch, obs = restore(env, duration=None)
            branch.fixed_delay = contrast['delay']
            policy = policies[contrast['failed_policy']]
            reference = next(r for r in anchor['policies'][contrast['failed_policy']]['episodes']
                             if r['delay'] == contrast['delay'])
            expected = {h['frame']: h for h in reference['history']}
            for target in targets:
                while branch.frame < target:
                    frame = branch.frame
                    obs, _, done, _, _ = branch.step(predict(policy, obs))
                    if done or physical_trace(branch.history[-1:]) != physical_trace([expected[frame]]):
                        raise AssertionError('Reference differs')
                rows = []
                baseline = expected[target]['request']
                actions = sorted({c.route.actions[0] for c in branch.choices})
                for action in actions:
                    start = time.perf_counter()
                    score = forecast(branch, action, geometric,
                                     policy=policy if args.tail == 'ppo' else None)
                    measured = next((r for r in contrast['trials'] if r['frame'] == target
                                     and r['request'] == action), None)
                    rows.append({'action': action, **score, 'ms': 1000 * (time.perf_counter() - start),
                                 'recorded_trial': measured})
                chosen = max(rows, key=lambda r: (r['worst_frames'], r['action'] == baseline, -r['action']))
                cases.append({'source': contrast['source'], 'delay': contrast['delay'], 'frame': target,
                              'baseline': baseline, 'selected': chosen['action'], 'scores': rows})
                print(contrast['source'], contrast['delay'], target, 'baseline/selected', baseline,
                      chosen['action'], [(r['action'], r['worst_frames']) for r in rows], flush=True)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'horizon': 60, 'drop_fraction': .5,
                   'tail': args.tail,
                   'candidate_prior': 'Zero actor prior for both tail controllers.',
                   'observation_model': 'Projected geometry, queue and candidates; other player fields fixed.',
                   'delay_model': 'Fixed zero/one-frame rollout branches with robust candidate checks.',
                   'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
