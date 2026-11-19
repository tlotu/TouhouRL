"""Check whether a cheap primitive screen preserves closed-loop rollout choices."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import HOLD, SHORT_HORIZON, ReplayWorker, hold_outcomes, rebase
from agent.short_planner import VECTORS
from scripts.diagnose_replay_mixed_windows import predict
from scripts.evaluate_replay_reselection import CountState, primitive_profile
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def inspect(episode, initial, buffer, case, worker):
    result = case['controller']
    events = {event['frame']: event for event in result['events']}
    belief, previous_count = initial, None
    rows = []
    for hold in result['history']:
        frame, request = hold['frame'], hold['request']
        sample = episode.samples[frame]
        position = tuple(hold['start'])
        scene = replace(sample.scene, player=position)
        if buffer not in belief:
            raise AssertionError('Reconstructed queue is outside the observed belief')
        obs = rebase(sample.observation, position)
        state = CountState(scene, belief, tuple(feasible_actions(obs, HOLD)))
        if frame in events:
            started = perf_counter()
            event = events[frame]
            horizon = 60 - HOLD if event['post_hold'] else 60
            count, scores = primitive_profile(state, horizon)
            last_count = previous_count
            if last_count is None and previous is not None:
                last_count = primitive_profile(previous, SHORT_HORIZON)[0]
            trigger = bool(hold['danger'] and last_count and count <= .5 * last_count)
            nominal, selected = event['nominal'], event['selected']
            elapsed_ms = (perf_counter() - started) * 1000
            choices, _, _ = worker.options(scene, obs, belief, waypoint_coverage=True)
            offered = {c.route.actions[0] for c in choices}
            full_scores = {int(action): value for action, value in event['mean_frames'].items()}
            if not set(full_scores).issubset(offered) or {nominal, selected} - offered:
                raise AssertionError('Recorded directions are missing from reconstructed candidates')
            ranked = sorted(offered, key=lambda a: (-scores[a], a != nominal, a))
            shortlist = set(ranked[:4]) | {nominal}
            scored = set(full_scores) & shortlist
            cached = (max(scored, key=lambda a: (full_scores[a], a == nominal, -a))
                      if nominal in scored else nominal)
            rows.append({'frame': frame, 'nominal': nominal, 'rollout_selected': selected,
                         'primitive_rank': ranked.index(selected) + 1,
                         'ranked_actions': ranked, 'primitive_scores': [float(scores[a]) for a in ranked],
                         'primitive_trigger': trigger, 'primitive_routes': count,
                         'previous_primitive_routes': last_count,
                         'shortlist_reproduces_choice': cached == selected,
                         'candidate_forecasts': len(offered) if len(offered) > 1 else 0,
                         'shortlist_forecasts': len(shortlist) if len(offered) > 1 else 0,
                         'extra_ms': elapsed_ms})
            previous_count = count
        else:
            previous_count = None
        previous = state
        moves, buffer = buffer.advance(request, HOLD, case['delay'])
        elapsed = min(HOLD, result['through_frame'] - frame)
        endpoint = position + scene.speed * VECTORS[list(moves[:elapsed])].sum(axis=0)
        np.testing.assert_allclose(endpoint, hold['end'], atol=1e-7, rtol=0)
        if elapsed == HOLD:
            belief = frozenset(hold_outcomes(scene, belief, request)[tuple(np.round(endpoint, 6))])
    if len(rows) != len(events):
        raise AssertionError('Some events had no executed hold')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    old_path = Path('runs/hrl-young-replay-smoke-001/manager.zip')
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, old_path, args.input]}
    report = json.loads(args.input.read_text(encoding='utf-8'))
    if any(report['sources'].get(str(p)) != sources[str(p)] for p in [BASE, *TRAIN, old_path]):
        raise ValueError('Replay sources differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policy = PPO.load(old_path, device='cpu')
    cases = []
    anchors = {}
    for case in report['cases']:
        base = case['baseline']
        key = (base['source'], base['episode'], case['anchor'])
        if key not in anchors:
            index = next(i for i, ep in enumerate(episodes) if (ep.source, ep.number) == key[:2])
            env = CoverageEnv(episodes, worker, verify=True)
            obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
            while env.frame < key[-1]:
                obs, _, done, _, _ = env.step(predict(policy, obs))
                if done:
                    raise AssertionError('Stopped before anchor')
            np.testing.assert_allclose(env.position, case['controller']['history'][0]['start'], atol=1e-7, rtol=0)
            anchors[key] = env.episode, env.belief, env.buffer
        rows = inspect(*anchors[key], case, worker)
        cases.append({'source': base['source'], 'episode': base['episode'], 'policy': case['policy'],
                      'delay': case['delay'], 'events': rows})
        print(key, case['policy'], case['delay'],
              [(r['frame'], r['primitive_rank'], r['primitive_trigger']) for r in rows], flush=True)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'cases': cases,
                   'scope': 'Shadow rank among all offered actions, then shortlist comparison using cached rollout scores.'},
                  stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
