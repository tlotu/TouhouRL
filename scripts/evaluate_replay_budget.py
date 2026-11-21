"""Compare bounded first-action search with ordinary and full-teacher ordering."""
import argparse
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import Choice, HOLD, PADDING, ReplayWorker, rebase, route_motions
from agent.short_planner import ANGLES, Route, TTC_HORIZON, VECTORS
from agent.swept_collision import SweptCollision, UniqueSweeps
from scripts.diagnose_replay_mixed_windows import predict
from scripts.evaluate_replay_reselection import drive
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, attack_position_summary, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


REFERENCE = Path('runs/hrl-young-batch-culling-profile-001.json')
MANAGERS = {name: Path('runs', folder, 'manager.zip') for name, folder in (
    ('old', 'hrl-young-replay-smoke-001'), ('paired', 'hrl-young-delay-paired-001'))}


@lru_cache(maxsize=16384)
def offsets(actions, belief):
    moves = route_motions(actions, belief)
    return VECTORS[np.asarray(moves)].cumsum(axis=1)


def verify_layer(scene, model, belief, sequences, parents, final):
    # Every parent cleared all previous holds. Verify the new hold across all
    # delay histories, retaining the worst clearance and terminal stopping TTC.
    paths = [offsets(seq, belief) for seq in sequences]
    lengths = [len(path) for path in paths]
    origins = np.cumsum([0, *lengths[:-1]])
    frame = (len(sequences[0]) - 1) * HOLD
    vertices = np.concatenate([path[:, max(0, frame - 1):] for path in paths], axis=0)
    points = scene.player + scene.speed * vertices
    start = points[:, 0] if frame else np.broadcast_to(scene.player, (len(points), 2))
    gap = np.full(len(points), model.clearance_cap)
    valid = np.ones(len(points), dtype=bool)
    for offset, end in enumerate(points[:, int(bool(frame)):].transpose(1, 0, 2)):
        clear, distance = model.edges(start, end, frame + offset, 1)
        valid &= clear
        gap = np.minimum(gap, distance)
        start = end
    good = np.logical_and.reduceat(valid, origins)
    gaps = np.minimum.reduceat(gap, origins)
    times = np.zeros(len(start))
    if final and good.any():
        active = np.repeat(good, lengths)
        times[active] = model.stationary_ttc(start[active], frame + HOLD, TTC_HORIZON)
    ttcs = np.minimum.reduceat(times, origins)
    best = {}
    for index in np.flatnonzero(good):
        seq = sequences[index]
        previous_gap, turn, heading = parents[seq[:-1]]
        gap = min(previous_gap, float(gaps[index]))
        turn += float(ANGLES[heading, seq[-1]])
        ttc = float(ttcs[index])
        score = gap + .25 * ttc - .5 * turn
        if seq[0] not in best or score > best[seq[0]][0]:
            best[seq[0]] = (score, seq, gap, turn, seq[-1] or heading, ttc)
    return best


def small_search(scene, belief, ranking, depth):
    model = SweptCollision(replace(scene, player_radius=scene.player_radius + PADDING))
    state = min(belief, key=lambda q: (q.requested, q.applied, q.pending))
    parents = {(): (model.clearance_cap, 0., state.requested)}
    sequences = [(action,) for action in ranking]
    checked = 0
    for layer in range(depth):
        checked += len(sequences)
        best = verify_layer(scene, model, belief, sequences, parents, layer + 1 == depth)
        if not best:
            return (), checked
        parents = {seq: (gap, turn, heading) for _, seq, gap, turn, heading, _ in best.values()}
        if layer + 1 < depth:
            sequences = [seq + (move,) for seq in parents for move in range(len(VECTORS))]
    # Choose the first verified proposal in the supplied order in both arms.
    for action in ranking:
        if action in best:
            _, seq, gap, turn, _, ttc = best[action]
            point = scene.player + scene.speed * offsets(seq, belief)[0, -1]
            route = Route((), tuple(point), seq, turn=turn)
            return (Choice(route, gap, ttc, point),), checked


class BudgetEnv(CoverageEnv):
    def bounded(self, teacher):
        start = perf_counter()
        obs = rebase(self.episode.samples[self.frame].observation, self.position)
        prior = self.worker.logits(obs)
        allowed = feasible_actions(obs, HOLD)
        ordinary = sorted(allowed, key=lambda a: (-prior[a], a))
        ranking = ordinary.copy()
        if teacher in ranking:
            ranking.remove(teacher)
            ranking.insert(0, teacher)
        ranking = ranking[:self.width]
        if self.set_only:
            ranking.sort(key=ordinary.index)
        choices, edges = small_search(self.scene, self.belief, ranking, self.depth)
        ms = (perf_counter() - start) * 1000
        self.audit.append(dict(frame=self.frame, teacher=teacher, ordinary=ordinary[:self.width],
                               shortlist=ranking, chosen=choices[0].route.actions[0] if choices else None,
                               extensions=edges, ms=ms))
        return choices, edges, ms

    def _plan_options(self, obs):
        if self.online_teacher:
            return super()._plan_options(obs)
        row = self.reference.get(self.frame) if self.use_teacher else None
        if row and not np.array_equal(self.position, row['start']):
            raise AssertionError('Cached teacher prefix changed')
        return self.bounded(row['request'] if row else None)

    def step(self, slot):
        if self.online_teacher:
            teacher = self.choices[slot].route.actions[0] if self.choices else None
            self.choices, _, _ = self.bounded(teacher)
            slot = 0
        return super().step(slot)


def evaluate(anchor, case, policy, forecast_worker, width, depth, teacher, online=False, set_only=False):
    env, _ = restore(anchor, duration=None)
    env.__class__ = BudgetEnv
    env.fixed_delay = case['delay']
    env.width, env.depth = width, depth
    env.use_teacher, env.online_teacher, env.set_only = teacher, online, set_only
    env.reference = {r['frame']: r for r in case['controller']['history']}
    env.audit = []
    env.previous_capacity = anchor.previous_capacity
    obs = env._prepare()
    if online:
        result = drive(env, obs, policy, forecast_worker, stop_at_horizon=True, prune_forecast_branches=True)
    else:
        while True:
            row = env.audit[-1]
            if teacher and row['chosen'] != row['teacher'] and row['chosen'] is not None:
                # A different request invalidates subsequent prerecorded advice.
                return evaluate(anchor, case, policy, forecast_worker, width, depth, True, True, set_only)
            _, _, done, _, result = env.step(0)
            if done:
                break
    result.update(source=env.episode.source, episode=env.episode.number)
    return {**{key: result[key] for key in ('source', 'episode', 'outcome', 'frames', 'min_clearance_px',
                                             'first_below_padding_frame', 'history')},
            'attack': attack_position_summary(env.episodes, [result]), 'teacher_recomputed': online, 'set_only': set_only,
            'width': width, 'depth': depth, 'teacher': teacher, 'policy': case['policy'],
            'delay': case['delay'], 'start': anchor.frame, 'end': env.frame, 'decisions': env.audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--set-only', action='store_true',
                        help='Isolate shortlist membership at the smallest budget, preserving PPO order.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists')
    torch.set_num_threads(1)
    report = json.loads(REFERENCE.read_text())
    sources = report['sources']
    if any(digest(Path(p)) != sha for p, sha in sources.items()):
        raise ValueError('Teacher inputs changed')
    episodes = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    projected = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True, collision_factory=UniqueSweeps)
    policies = {k: PPO.load(p, device='cpu') for k, p in MANAGERS.items()}
    cases = []
    for index, episode in enumerate(episodes):
        group = [c for c in report['cases'] if (episode.source, episode.number) ==
                 (c['controller']['source'], c['controller']['episode'])]
        if not group:
            continue
        anchor = CoverageEnv(episodes, worker, verify=True)
        obs, _ = anchor.reset(seed=0, options={'episode': index, 'delay': 0})
        while anchor.frame < group[0]['anchor']:
            obs, _, done, _, _ = anchor.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Anchor unreachable')
        for case in group:
            budgets = ((2, 2),) if args.set_only else ((2, 2), (3, 2), (2, 3), (3, 3))
            for width, depth in budgets:
                for teacher in ((True,) if args.set_only else (False, True)):
                    row = evaluate(anchor, case, policies[case['policy']], projected, width, depth, teacher, set_only=args.set_only)
                    cases.append(row)
                    print(index, row['policy'], row['delay'], width, depth, teacher,
                          row['outcome'], row['end'], flush=True)
    if len(cases) != (8 if args.set_only else 64):
        raise AssertionError('Incomplete comparison')
    with args.output.open('x') as stream:
        json.dump(dict(sources=sources, teacher_report=digest(REFERENCE), cases=cases), stream, allow_nan=False)


if __name__ == '__main__':
    main()

