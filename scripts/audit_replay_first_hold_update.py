"""Compare predicted and refreshed continuation after one executed hold."""
import argparse
from copy import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from agent.replay_async import project
from agent.replay_hrl import HOLD, ReplayWorker, hold_outcomes
from scripts.audit_replay_replanning_scores import forecast
from scripts.audit_training_escape import fork_env, physical_trace
from scripts.diagnose_replay_mixed_windows import predict, rollout
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


def mean_frames(score):
    frames = [branch['frames'] for branch in score['branches']]
    return sum(frames) / len(frames)


def slot_for(env, request):
    return next(i for i, choice in enumerate(env.choices) if choice.route.actions[0] == request)


def outcome(env, observation, policy, first_slot=None):
    branch = fork_env(env)
    result = rollout(branch, observation, policy, first_slot=first_slot)
    return {'through_frame': branch.frame, 'outcome': result['outcome'],
            'min_clearance_px': result['min_clearance_px'],
            'first_below_padding_frame': result['first_below_padding_frame']}


def inspect_hold(origin, policy, worker, prior, horizon):
    selected = max(prior['scores'], key=lambda r: (mean_frames(r), r['action'] == prior['baseline'], -r['action']))
    action = selected['action']
    sample = origin.episode.samples[origin.frame]
    branch = fork_env(origin)
    branch.min_gap = float('inf')
    branch.padded_contact = branch.last_padded_contact = None
    obs, _, done, _, info = branch.step(slot_for(branch, action))
    row = {'frame': origin.frame, 'delay': origin.fixed_delay, 'source': origin.episode.source,
           'episode': origin.episode.number, 'baseline': prior['baseline'], 'request': action,
           'prior_branches': selected['branches'], 'prior_mean_frames': mean_frames(selected),
           'arrived_frame': branch.frame, 'arrived_position': branch.position.tolist(),
           'first_hold_min_clearance_px': info['min_clearance_px'], 'first_hold_outcome': info['outcome']}
    if done:
        return row
    endpoint = tuple(np.round(branch.position, 6))
    outcomes = hold_outcomes(origin.scene, origin.belief, action)
    if branch.belief != frozenset(outcomes[endpoint]):
        raise AssertionError('Arrival did not match the queue update')
    row['belief_after'] = [asdict(q) for q in sorted(branch.belief, key=lambda q: (q.requested, q.applied, q.pending))]
    row['predicted_endpoint_count'] = len(outcomes)
    row['arrival_directions'] = sorted({choice.route.actions[0] for choice in branch.choices})

    projected = copy(branch)
    projected.scene = project(origin.scene, tuple(branch.position))
    row['projected_update'] = forecast(projected, None, worker, horizon, policy, sample=sample)
    row['observed_update'] = forecast(branch, None, worker, horizon, policy)
    row['ppo_continuation'] = outcome(branch, obs, policy)

    if branch.choices:
        nominal_slot = predict(policy, obs)
        nominal = branch.choices[nominal_slot].route.actions[0]
        scores = [{'action': move, **forecast(branch, move, worker, horizon, policy)}
                  for move in row['arrival_directions']]
        # The reference worker has a learned prior. Preserve PPO's action if the
        # zero-prior forecast cannot evaluate that action on equal terms.
        available = [r for r in scores if all(b['frames'] >= HOLD for b in r['branches'])]
        reranked = nominal
        if any(r['action'] == nominal for r in available):
            reranked = max(available, key=lambda r: (mean_frames(r), r['action'] == nominal, -r['action']))['action']
        row['reselection'] = {'nominal': nominal, 'selected': reranked, 'scores': scores}
        row['reselected_continuation'] = (row['ppo_continuation'] if reranked == nominal else
                                          outcome(branch, obs, policy, slot_for(branch, reranked)))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    anchor_path = Path('runs/hrl-young-delay-anchors-001.json')
    forecast_path = Path('runs/hrl-young-replanning-ppo-scores-001.json')
    paths = {'old': Path('runs/hrl-young-replay-smoke-001/manager.zip'),
             'paired': Path('runs/hrl-young-delay-paired-001/manager.zip')}
    anchors = json.loads(anchor_path.read_text(encoding='utf-8'))
    prior = json.loads(forecast_path.read_text(encoding='utf-8'))
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, anchor_path, forecast_path, *paths.values()]}
    if any(prior['sources'].get(str(p)) != sources[str(p)] for p in [BASE, *TRAIN, anchor_path, *paths.values()]):
        raise ValueError('Previous forecasts use different sources')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    actual = ReplayWorker(backend.action_logits, prune_failed=True)
    projected = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True)
    policies = {name: PPO.load(p, device='cpu') for name, p in paths.items()}
    cases = []
    remaining = prior['horizon'] - HOLD
    for anchor in anchors['anchors']:
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        env = CoverageEnv(episodes, actual, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Stopped before anchor')
        for name, policy in policies.items():
            for reference in anchor['policies'][name]['episodes']:
                if reference['outcome'] != 'no_verified_route':
                    continue
                trials = sorted((c for c in prior['cases'] if (c['source'], c['delay']) ==
                                 (anchor['source'], reference['delay'])), key=lambda c: c['frame'])
                branch, obs = restore(env, duration=None)
                branch.fixed_delay = reference['delay']
                expected = {row['frame']: row for row in reference['history']}
                for case in trials:
                    while branch.frame < case['frame']:
                        frame = branch.frame
                        obs, _, done, _, _ = branch.step(predict(policy, obs))
                        if done or physical_trace(branch.history[-1:]) != physical_trace([expected[frame]]):
                            raise AssertionError('Reference trajectory differs')
                    row = inspect_hold(branch, policy, projected, case, remaining)
                    cases.append(row)
                    print(row['source'], row['delay'], row['frame'], 'first move', row['request'],
                          'prior', row['prior_mean_frames'], 'projected / refreshed remaining',
                          mean_frames(row['projected_update']) if 'projected_update' in row else None,
                          mean_frames(row['observed_update']) if 'observed_update' in row else None,
                          'PPO / reselected', row.get('ppo_continuation'), row.get('reselected_continuation'), flush=True)
    if len(cases) != len(prior['cases']):
        raise AssertionError('Missing or repeated decision case')
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'restart_horizon_frames': remaining,
                   'method': 'One observed hold followed by matched-position and matched-belief forecasts.',
                   'future_delay': 'Equal-weight fixed zero/one-frame forecasts after receipt of the endpoint.',
                   'reselection': 'One additional mean-ranked hold, then normal PPO.',
                   'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
