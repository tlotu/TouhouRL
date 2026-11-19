"""Exhaustively check short routes at replay stops with no verified option."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.forward_model import ForwardModel
from agent.replay_hrl import HOLD, PADDING, SHORT_HORIZON, Choice, ReplayWorker, rebase
from agent.short_planner import Route, VECTORS
from scripts.audit_training_escape import fork_env, physical_trace
from scripts.diagnose_replay_mixed_windows import predict, rollout
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


def feasible_prefixes(scene, queues, delays, padding, observations=None, *, model_factory=ForwardModel,
                      final_counts=False, batch_size=128):
    """Enumerate every request sequence, rejecting only unsafe movement branches."""
    model = model_factory(replace(scene, player_radius=scene.player_radius + padding))
    models = ([(model_factory(replace(sample, player_radius=sample.player_radius + padding)), offset)
               for sample, offset in observations] if observations is not None
              else [(model, frame) for frame in range(SHORT_HORIZON)])
    frontier = [((), [(scene.player, queue) for queue in queues], float('inf'))]
    for elapsed in range(0, len(models), HOLD):
        states = dict.fromkeys(state for _, endpoints, _ in frontier for state in endpoints)
        requests = [(state, action) for state in states for action in range(len(VECTORS))]
        transitions = {}
        for begin in range(0, len(requests), batch_size):
            batch = requests[begin:begin + batch_size]
            branches = [(point, moves, tail) for (point, queue), action in batch for delay in delays
                        for moves, tail in (queue.advance(action, HOLD, delay),)]
            starts = np.asarray([point for point, _, _ in branches])
            moves = np.asarray([moves for _, moves, _ in branches])
            legal = np.ones(len(branches), dtype=bool)
            clearances = np.full(len(branches), np.inf)
            for offset in range(HOLD):
                ends = starts + scene.speed * VECTORS[moves[:, offset]]
                collision, time = models[elapsed + offset]
                safe, distances = collision.edges(starts, ends, time, 1)
                legal &= safe
                clearances = np.minimum(clearances, distances)
                starts = ends
            for index, request in enumerate(batch):
                lo = index * len(delays)
                hi = lo + len(delays)
                if legal[lo:hi].all():
                    endpoints = [(tuple(starts[i]), branches[i][2]) for i in range(lo, hi)]
                    transitions[request] = endpoints, float(clearances[lo:hi].min())
        if final_counts and elapsed + HOLD == len(models):
            # Each bit is one safe request. Intersect across possible input states.
            masks = {state: sum(1 << a for a in range(len(VECTORS)) if (state, a) in transitions)
                     for state in states}
            counts = [0] * len(VECTORS)
            for prefix, endpoints, _ in frontier:
                mask = (1 << len(VECTORS)) - 1
                for endpoint in endpoints:
                    mask &= masks[endpoint]
                if prefix:
                    counts[prefix[0]] += mask.bit_count()
                else:
                    for action in range(len(VECTORS)):
                        counts[action] += (mask >> action) & 1
            yield elapsed + HOLD, counts
            return
        children = []
        for prefix, states, gap in frontier:
            for action in range(len(VECTORS)):
                edges = [transitions.get((state, action)) for state in states]
                if any(edge is None for edge in edges):
                    continue
                endpoints = list(dict.fromkeys(point for edge, _ in edges for point in edge))
                clearance = min(gap, min(value for _, value in edges))
                children.append((prefix + (action,), endpoints, clearance))
        frontier = children
        yield elapsed + HOLD, [(actions, gap + padding) for actions, _, gap in frontier]


def probe(env):
    obs = rebase(env.episode.samples[env.frame].observation, env.position)
    allowed = feasible_actions(obs, HOLD)
    layers = []
    trials = {}
    for mode, queues, delays in (
        ('belief', env.belief, (0, 1)),
        ('known_queue', (env.buffer,), (0, 1)),
        ('fixed_delay', (env.buffer,), (env.fixed_delay,)),
    ):
        for padding in (PADDING, 0.):
            for frames, paths in feasible_prefixes(env.scene, queues, delays, padding):
                masked = [(actions, gap) for actions, gap in paths if actions[0] in allowed]
                layers.append({'input': mode, 'margin_px': padding, 'frames': frames,
                               'routes': len(paths), 'masked_routes': len(masked),
                               'directions': sorted({actions[0] for actions, _ in paths}),
                               'masked_directions': sorted({actions[0] for actions, _ in masked})})
                for actions, gap in paths:
                    key = (mode == 'belief', mode == 'known_queue', padding,
                           len(actions), actions[0] in allowed, gap)
                    previous = trials.get(actions[0])
                    if previous is None or key > previous[0]:
                        trials[actions[0]] = (key, mode, padding, actions, gap)
    return allowed, layers, trials


def refreshed_probe(env):
    frames = min(SHORT_HORIZON, (env.limit - env.frame) // HOLD * HOLD)
    samples = env.episode.samples
    observations = []
    for frame in range(env.frame, env.frame + frames):
        sampled_at = frame if frame in samples else frame - 1
        observations.append((samples[sampled_at].scene, frame - sampled_at))
    layers = []
    for mode, queues, delays in (('belief', env.belief, (0, 1)),
                                ('fixed_delay', (env.buffer,), (env.fixed_delay,))):
        for padding in (PADDING, 0.):
            for elapsed, paths in feasible_prefixes(env.scene, queues, delays, padding, observations):
                best = max(paths, key=lambda path: path[1], default=None)
                layers.append({'input': mode, 'margin_px': padding, 'frames': elapsed,
                               'routes': len(paths), 'best_route': best[0] if best else None,
                               'best_gap_px': best[1] if best and np.isfinite(best[1]) else None})
    return layers


def try_hold(origin, policy, trial, allowed):
    _, mode, padding, actions, gap = trial
    env = fork_env(origin)
    env.min_gap = float('inf')
    env.padded_contact = env.last_padded_contact = None
    # The audit independently checked this route under the recorded assumptions.
    # Run only its first hold, then reinstate the normal verified worker.
    target = np.asarray(env.position)
    actual_moves, _ = env.buffer.advance(actions[0], HOLD, env.fixed_delay)
    target = target + env.scene.speed * VECTORS[list(actual_moves)].sum(axis=0)
    env.choices = (Choice(Route((), tuple(target), actions), gap - padding, 0., target),)
    standard = (mode == 'belief' and padding == PADDING
                and len(actions) * HOLD == SHORT_HORIZON and actions[0] in allowed)
    env.verify = standard
    obs, _, done, _, result = env.step(0)
    first_hold = {'through_frame': env.frame, 'position': env.position.tolist(),
                  'outcome': result['outcome'], 'gap_px': result['min_clearance_px'],
                  'normal_choices': len(env.choices) if not done else None}
    env.verify = True
    if not done:
        result = rollout(env, obs, policy)
    return {'input': mode, 'margin_px': padding, 'route': actions,
            'frozen_gap_px': gap if np.isfinite(gap) else None,
            'standard_constraints': standard, 'first_hold': first_hold,
            'through_frame': env.frame, 'outcome': result['outcome'],
            'first_below_padding_frame': result['first_below_padding_frame'],
            'min_clearance_px': result['min_clearance_px']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    paths = {'old': args.old, 'paired': args.paired}
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, *paths.values(), args.anchors]}
    report = json.loads(args.anchors.read_text(encoding='utf-8'))
    if any(report['sources'].get(str(p)) != sources[str(p)] for p in [BASE, *TRAIN, *paths.values()]):
        raise ValueError('Anchor sources differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    policies = {name: PPO.load(path, device='cpu') for name, path in paths.items()}
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    cases = []
    for anchor in report['anchors']:
        failures = [(name, row) for name in policies for row in anchor['policies'][name]['episodes']
                    if row['outcome'] == 'no_verified_route']
        if not failures:
            continue
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        baseline = CoverageEnv(episodes, worker, verify=True)
        obs, _ = baseline.reset(seed=0, options={'episode': index, 'delay': 0})
        while baseline.frame < anchor['frame']:
            obs, _, done, _, _ = baseline.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Stopped before anchor')
        for name, expected in failures:
            env, obs = restore(baseline, duration=None)
            env.fixed_delay = expected['delay']
            result = rollout(env, obs, policies[name])
            through = expected['start'] + expected['frames']
            if (env.frame != through or result['outcome'] != expected['outcome']
                    or physical_trace(result['history']) != physical_trace(expected['history'])):
                raise AssertionError('Terminal trajectory did not reproduce')
            allowed, layers, trials = probe(env)
            refreshed = refreshed_probe(env)
            print(env.episode.source, name, env.fixed_delay, through, flush=True)
            print('Frozen 16-frame counts:', [(r['input'], r['margin_px'], r['routes'])
                                              for r in layers if r['frames'] == SHORT_HORIZON], flush=True)
            print('Refreshed 16-frame counts:', [(r['input'], r['margin_px'], r['routes'])
                                                 for r in refreshed if r['frames'] == SHORT_HORIZON], flush=True)
            branches = [try_hold(env, policies[name], trial, allowed)
                        for _, trial in sorted(trials.items())]
            print('First hold trials:', [(r['route'][0], r['through_frame'], r['outcome'])
                                         for r in branches], flush=True)
            cases.append({'source': env.episode.source, 'episode': env.episode.number,
                          'policy': name, 'delay': env.fixed_delay, 'frame': through,
                          'position': env.position.tolist(), 'input_buffer': asdict(env.buffer),
                          'belief': [asdict(q) for q in sorted(env.belief,
                                     key=lambda q: (q.requested, q.applied, q.pending))],
                          'boundary_mask': allowed, 'layers': layers, 'refreshed_layers': refreshed,
                          'closed_loop': branches})
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
