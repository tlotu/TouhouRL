"""Evaluate automatic contraction-triggered reselection in closed-loop replay."""
import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import HOLD, PADDING, SHORT_HORIZON, ReplayWorker, rebase
from agent.short_planner import VECTORS
from agent.swept_collision import SweptCollision, UniqueSweeps
from scripts.audit_replay_first_hold_update import mean_frames, slot_for
from scripts.audit_replay_replanning_scores import forecast_candidates
from scripts.audit_replay_terminal_routes import feasible_prefixes
from scripts.audit_training_escape import physical_trace
from scripts.diagnose_replay_mixed_windows import predict
from scripts.profile_replay_options import PhaseTimes
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, attack_position_summary, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


@dataclass(frozen=True)
class CountState:
    scene: object
    belief: frozenset
    allowed: tuple

    @classmethod
    def from_env(cls, env):
        observation = env.episode.samples[env.frame].observation
        return cls(env.scene, env.belief, tuple(feasible_actions(rebase(observation, env.position), HOLD)))


def count_routes(state):
    # The cap clips reported clearances, but leaves collision validity unchanged.
    for _, counts in feasible_prefixes(state.scene, state.belief, (0, 1), PADDING,
                                      final_counts=True, batch_size=1024, prune_failed=True, reuse_edges=True,
                                      vector_terminal=True,
                                      model_factory=lambda scene: SweptCollision(
                                          scene, clearance_cap=1e-6, cull_batches=True)):
        pass
    return sum(counts[a] for a in state.allowed)


@lru_cache(maxsize=256)
def primitive_motions(belief):
    """All first requests followed by one sustained direction, with both delays."""
    count = len(VECTORS)
    motions = []
    for first, second, initial_delay, later_delay in product(range(count), range(count), (0, 1), (0, 1)):
        for queue in belief:
            prefix, tail = queue.advance(first, HOLD, initial_delay)
            suffix, _ = tail.advance(second, SHORT_HORIZON - HOLD, later_delay)
            motions.append(prefix + suffix)
    return np.asarray(motions)


def primitive_profile(state, horizon):
    scene = state.scene
    model = SweptCollision(replace(scene, player_radius=scene.player_radius + PADDING), clearance_cap=1e-6)
    motions = primitive_motions(state.belief)
    starts = np.broadcast_to(scene.player, motions.shape[:1] + (2,)).copy()
    duration = np.zeros(len(motions))
    alive = np.arange(len(motions))
    for frame in range(SHORT_HORIZON):
        ends = starts + scene.speed * VECTORS[motions[alive, frame]]
        safe, _ = model.edges(starts, ends, frame, 1)
        starts, alive = ends[safe], alive[safe]
        duration[alive] = frame + 1
        if not len(alive):
            break
    count = len(VECTORS)
    shape = (count, count, 2, 2, len(state.belief))
    valid = (duration.reshape(shape) == SHORT_HORIZON).all(axis=(2, 3, 4))
    routes = int(valid[list(state.allowed)].sum())
    if horizon > SHORT_HORIZON and len(alive):
        duration[alive] += model.stationary_ttc(starts, SHORT_HORIZON, horizon - SHORT_HORIZON)
    # After the first hold, an observed endpoint may support a different
    # continuation. Keep both first-delay scores and average them only at ranking.
    scores = duration.reshape(shape).min(axis=(3, 4)).max(axis=1).mean(axis=1)
    return routes, scores


def choose(env, policy, worker, nominal, horizon, actions=None, *, stop_at_horizon=False, pruned=None):
    if actions is None:
        actions = sorted({choice.route.actions[0] for choice in env.choices})
    if len(actions) == 1:
        return nominal, {}, 0
    if stop_at_horizon or pruned is not None:
        actions = [nominal, *(a for a in actions if a != nominal)]
    max_frames = ((horizon + HOLD - 1) // HOLD) * HOLD
    scores = {}
    calls = 0

    def rank(action, score):
        return score, action == nominal, -action

    def can_win(action, ceiling):
        if action == nominal or nominal not in scores:
            return True
        if rank(action, ceiling) > max(rank(a, s) for a, s in scores.items()):
            return True
        pruned[action] = ceiling
        return False

    kwargs = {'can_win': can_win} if pruned is not None else {}
    for action, result in forecast_candidates(env, actions, worker, horizon, policy, **kwargs):
        calls += 1
        if result is None:
            continue
        if all(branch['frames'] >= HOLD for branch in result['branches']):
            scores[action] = mean_frames(result)
            if stop_at_horizon and action == nominal and scores[action] == max_frames:
                # The nominal action wins ties and no branch can exceed this score.
                return nominal, scores, calls
    if nominal not in scores:
        return nominal, scores, calls
    return max(scores, key=lambda action: rank(action, scores[action])), scores, calls


def drive(env, obs, policy, worker, method='rollout', *, profile=False, stop_at_horizon=False,
          prune_forecast_branches=False):
    previous = previous_count = None
    followup = False
    events, decision_ms = [], []
    checked = 0
    phase_rows = []
    timer = PhaseTimes() if profile else None
    count = timer.timed(count_routes, 'exact_count') if timer else count_routes
    score_primitives = timer.timed(primitive_profile, 'primitive') if timer else primitive_profile
    while True:
        started = perf_counter()
        slot = predict(policy, obs)
        current_count = None
        primitive_scores = None
        current = CountState.from_env(env) if env.choices else None
        trigger = False
        if current is not None and previous is not None and env.danger:
            checked += 1
            if method == 'primitive':
                current_count, primitive_scores = score_primitives(current, 60)
            else:
                current_count = count(current)
            if previous_count is None:
                previous_count = (score_primitives(previous, SHORT_HORIZON)[0]
                                  if method == 'primitive' else count(previous))
            trigger = previous_count > 0 and current_count <= previous_count * .5
        recheck = followup
        followup = False
        if current is not None and (trigger or recheck):
            nominal = env.choices[slot].route.actions[0]
            horizon = 60 - HOLD if recheck else 60
            offered = sorted({c.route.actions[0] for c in env.choices})
            shortlisted = offered
            forecast_calls = 0
            pruned = {} if prune_forecast_branches else None
            scoring_started = perf_counter()
            if method == 'primitive':
                if primitive_scores is None or recheck:
                    _, primitive_scores = score_primitives(current, horizon)
                scores = {c.route.actions[0]: float(primitive_scores[c.route.actions[0]]) for c in env.choices}
                action = max(scores, key=lambda move: (scores[move], move == nominal, -move))
            else:
                if method == 'hybrid' and len(offered) > 4:
                    _, primitive_scores = score_primitives(current, horizon)
                    ranked = sorted(offered, key=lambda move: (-primitive_scores[move], move != nominal, move))
                    shortlisted = sorted(set(ranked[:4]) | {nominal})
                if timer:
                    with timer.options(worker):
                        timed_choose = timer.timed(choose, 'forecast_selection')
                        action, scores, forecast_calls = timed_choose(
                            env, policy, worker, nominal, horizon, shortlisted,
                            stop_at_horizon=stop_at_horizon, pruned=pruned)
                else:
                    action, scores, forecast_calls = choose(
                        env, policy, worker, nominal, horizon, shortlisted,
                        stop_at_horizon=stop_at_horizon, pruned=pruned)
            scoring_ms = (perf_counter() - scoring_started) * 1000
            if action != nominal:
                slot = slot_for(env, action)
                followup = trigger and not recheck
            events.append({'frame': env.frame, 'trigger': trigger, 'post_hold': recheck,
                           'previous_routes': previous_count, 'current_routes': current_count,
                           'nominal': nominal, 'selected': action, 'mean_frames': scores,
                           'offered_actions': offered, 'shortlisted_actions': shortlisted,
                           'forecast_calls': forecast_calls,
                           'scoring_ms': scoring_ms,
                           'selection_ms': (perf_counter() - started) * 1000})
            if stop_at_horizon and 0 < forecast_calls < len(shortlisted):
                events[-1]['skipped_forecasts'] = [move for move in shortlisted if move != nominal]
            if pruned:
                events[-1]['pruned_forecasts'] = pruned
        decision_ms.append((perf_counter() - started) * 1000)
        if timer:
            phase_rows.append({'frame': env.frame, **timer.take()})
        previous, previous_count = current, current_count
        obs, _, done, _, result = env.step(slot)
        if done:
            if timer:
                result = {**result, 'decision_phases': phase_rows}
            return {**result, 'source': env.episode.source, 'episode': env.episode.number,
                    'through_frame': env.frame, 'route_count_decisions': checked,
                    'events': events, 'extra_decision_ms': decision_ms}


def compact(row, episodes):
    return {**{k: row[k] for k in ('source', 'episode', 'outcome', 'frames', 'first_below_padding_frame',
                                    'min_clearance_px')},
            'attack_position': attack_position_summary(episodes, [row])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-index', type=int, choices=range(len(TRAIN)))
    parser.add_argument('--policy', choices=('old', 'paired'))
    parser.add_argument('--method', choices=('rollout', 'primitive', 'hybrid'), default='rollout')
    parser.add_argument('--profile', action='store_true', help='Record per-decision component timings.')
    parser.add_argument('--stop-at-horizon', action='store_true',
                        help='Skip other forecasts when the nominal action reaches the scoring horizon.')
    parser.add_argument('--prune-forecast-branches', action='store_true',
                        help='Stop a candidate once its optimistic score cannot win.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    anchor_path = Path('runs/hrl-young-delay-anchors-001.json')
    paths = {'old': Path('runs/hrl-young-replay-smoke-001/manager.zip'),
             'paired': Path('runs/hrl-young-delay-paired-001/manager.zip')}
    report = json.loads(anchor_path.read_text(encoding='utf-8'))
    inputs = [BASE, *TRAIN, *paths.values()]
    sources = {str(p): digest(p) for p in [*inputs, anchor_path]}
    if any(report['sources'].get(str(p)) != sources[str(p)] for p in inputs):
        raise ValueError('Anchor sources differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    actual = ReplayWorker(backend.action_logits, prune_failed=True)
    projected = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True, collision_factory=UniqueSweeps)
    policies = {name: PPO.load(p, device='cpu') for name, p in paths.items()}
    cases = []
    for anchor in report['anchors']:
        if args.capture_index is not None and anchor['source'] != str(TRAIN[args.capture_index]):
            continue
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        env = CoverageEnv(episodes, actual, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Stopped before anchor')
        for name in policies:
            if args.policy is not None and name != args.policy:
                continue
            for reference in anchor['policies'][name]['episodes']:
                branch, obs = restore(env, duration=None)
                branch.fixed_delay = reference['delay']
                print('Start', anchor['source'], name, reference['delay'], flush=True)
                result = drive(branch, obs, policies[name], projected, args.method, profile=args.profile,
                               stop_at_horizon=args.stop_at_horizon,
                               prune_forecast_branches=args.prune_forecast_branches)
                baseline = compact(reference, episodes)
                changes = [event for event in result['events'] if event['selected'] != event['nominal']]
                first = changes[0]['frame'] if changes else branch.frame + 1
                expected = [row for row in reference['history'] if row['frame'] < first]
                if physical_trace(expected) != physical_trace([row for row in result['history'] if row['frame'] < first]):
                    raise AssertionError('Matched control path changed before intervention')
                cases.append({'policy': name, 'delay': branch.fixed_delay, 'anchor': env.frame,
                              'baseline': baseline, 'controller': {**result, **compact(result, episodes)}})
                print('Finish', baseline['outcome'], result['outcome'], branch.frame,
                      'checks', len(result['events']), 'changes', [(r['frame'], r['nominal'], r['selected']) for r in changes], flush=True)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    if not cases:
        raise ValueError('No matching anchor branches')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'drop_fraction': .5, 'danger_gate': True, 'method': args.method,
                   'trigger_forecast_frames': 60, 'followup_forecast_frames': 56,
                   'timing_excludes_normal_environment_step': True, 'stop_at_horizon': args.stop_at_horizon,
                   'prune_forecast_branches': args.prune_forecast_branches,
                   'cases': cases},
                  stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
