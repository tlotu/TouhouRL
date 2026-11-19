"""Compare frozen forecasts with refreshed hazards along executed replay paths."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.forward_model import ForwardModel
from agent.replay_hrl import HOLD, PADDING, ReplayWorker
from agent.short_planner import VECTORS
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv


def reconstruct(episode, history, buffer, delay, through):
    edges = {}
    for hold in history:
        frame = hold['frame']
        position = np.asarray(hold['start'], dtype=float)
        if edges and not np.allclose(position, edges[frame - 1][1], atol=1e-7, rtol=0):
            raise AssertionError('Discontinuous history')
        moves, buffer = buffer.advance(hold['request'], HOLD, delay)
        for offset, move in enumerate(moves[:min(HOLD, through - frame)]):
            end = position + episode.samples[frame].scene.speed * VECTORS[move]
            edges[frame + offset] = (position.copy(), end.copy())
            position = end
        if not np.allclose(position, hold['end'], atol=1e-7, rtol=0):
            raise AssertionError(f'Input reconstruction differs at {frame}')
    return edges


def check(model, edge, offset):
    safe, gap = model.edges([edge[0]], [edge[1]], offset, 1)
    return bool(safe[0]), float(gap[0])


def finite(value):
    return float(value) if np.isfinite(value) else None


def audit(episode, edges, origin, horizon):
    model = ForwardModel(episode.samples[origin].scene)
    rows = []
    for frame in range(origin, min(origin + horizon, max(edges) + 1)):
        sample_frame = frame if frame in episode.samples else frame - 1
        refreshed = ForwardModel(episode.samples[sample_frame].scene)
        safe, gap = check(model, edges[frame], frame - origin)
        fresh_safe, fresh_gap = check(refreshed, edges[frame], frame - sample_frame)
        rows.append({'frame': frame, 'frozen_safe': safe, 'refreshed_safe': fresh_safe,
                     'frozen_clearance': finite(gap), 'refreshed_clearance': finite(fresh_gap)})
    def first(predicate):
        return next((row for row in rows if predicate(row)), None)
    return {'frame': origin, 'requested_frames': horizon, 'checked_frames': len(rows),
            'censored': len(rows) < horizon,
            'frozen_min_clearance': finite(min(r['frozen_clearance'] if r['frozen_clearance'] is not None
                                               else float('inf') for r in rows)),
            'refreshed_min_clearance': finite(min(r['refreshed_clearance'] if r['refreshed_clearance'] is not None
                                                  else float('inf') for r in rows)),
            'first_frozen_contact': first(lambda r: not r['frozen_safe']),
            'first_refreshed_contact': first(lambda r: not r['refreshed_safe']),
            'first_frozen_padding_contact': first(lambda r: r['frozen_clearance'] is not None
                                                  and r['frozen_clearance'] <= PADDING),
            'first_refreshed_padding_contact': first(lambda r: r['refreshed_clearance'] is not None
                                                     and r['refreshed_clearance'] <= PADDING),
            'first_clearance_drift_over_2px': first(lambda r: r['frozen_clearance'] is not None
                                                    and r['refreshed_clearance'] is not None
                                                    and abs(r['frozen_clearance'] - r['refreshed_clearance']) > 2)}


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
    report = json.loads(args.anchors.read_text(encoding='utf-8'))
    if any(report['sources'].get(str(p)) != sources[str(p)] for p in [BASE, *TRAIN, args.old]):
        raise ValueError('Anchor source mismatch')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(args.old, device='cpu')
    cases = []
    for anchor in report['anchors']:
        index = next(i for i, ep in enumerate(episodes)
                     if ep.source == anchor['source'] and ep.number == anchor['episode'])
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(int(old.predict(obs, deterministic=True)[0]))
            if done:
                raise AssertionError('Stopped before anchor')
        for policy in ('old', 'paired'):
            for branch in anchor['policies'][policy]['episodes']:
                if not np.allclose(env.position, branch['history'][0]['start'], atol=1e-7, rtol=0):
                    raise AssertionError('Anchor position differs')
                through = branch['start'] + branch['frames']
                edges = reconstruct(env.episode, branch['history'], env.buffer, branch['delay'], through)
                origins = sorted(set(range(anchor['frame'], through, 32))
                                 | {f for f in (384, 388, 396, 604, (through - 16) // HOLD * HOLD)
                                    if f in edges})
                forecasts = [audit(env.episode, edges, frame, horizon)
                             for frame in origins for horizon in (16, 60, 96)]
                case = {'source': anchor['source'], 'episode': anchor['episode'],
                        'anchor': anchor['frame'], 'policy': policy, 'delay': branch['delay'],
                        'outcome': branch['outcome'], 'through_frame': through,
                        'reconstructed_frames': len(edges), 'forecasts': forecasts}
                cases.append(case)
                print(case['source'], policy, branch['delay'], through, branch['outcome'],
                      'frozen contacts', sum(r['first_frozen_contact'] is not None for r in forecasts),
                      'refreshed contacts', sum(r['first_refreshed_contact'] is not None for r in forecasts),
                      flush=True)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'contract': 'executed-path-frozen-vs-refreshed-v1',
                   'limitations': ['Recorded hazards do not react to counterfactual player motion.',
                                   'Stopped traces are censored; no-route is not a collision.',
                                   'No hazard identities are matched; drift does not prove new births.'],
                   'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
