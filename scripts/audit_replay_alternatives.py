"""Test each first direction at a recorded closed-loop danger trigger."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.forward_model import ForwardModel
from agent.replay_hrl import Choice, PADDING, ReplayManagerEnv, ReplayWorker, rebase, robust_routes
from agent.short_planner import HOLD, LONG_HORIZON, SHORT_HORIZON, VECTORS, plan, route_points
from agent.swept_collision import SweptCollision
from scripts.train_replay_hrl import BASE, TEST, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_supervised import event_horizon


def seek(scene, belief, allowed, action, horizon, delay, width):
    """Propose at fixed delay, with robust validation for the executable prefix."""
    padded = replace(scene, player_radius=scene.player_radius + PADDING)
    collision = SweptCollision(padded)
    state = min(belief, key=lambda q: (q.requested, q.applied, q.pending))
    result = plan(padded, previous_action=state.requested, width=width, smooth=False,
                  input_buffer=state, switch_delay=delay, collision_model=collision,
                  horizon_override=horizon, first_action=action, ttc_weight=.25)
    verified = []
    for route in result.candidates:
        if len(route.actions) * HOLD != horizon:
            continue
        prefix = replace(route, actions=route.actions[:SHORT_HORIZON // HOLD])
        verified.append((route, prefix))
    robust = robust_routes(padded, belief, tuple(prefix for _, prefix in verified), allowed, collision)
    eligible = {prefix.actions: (gap, ttc) for prefix, gap, ttc in robust}
    for route, prefix in sorted(verified, key=lambda pair: pair[0].ttc, reverse=True):
        if prefix.actions not in eligible:
            continue
        points = route_points(padded, route)
        reference = ForwardModel(padded)
        if not all(reference.edges([start], [end], frame, 1)[0][0]
                   for frame, (start, end) in enumerate(zip(points, points[1:]))):
            raise AssertionError('Predicted route failed reference verification')
        gap, ttc = eligible[prefix.actions]
        return route, Choice(prefix, float(gap), float(ttc), points[SHORT_HORIZON])
    return None, None


def branch(episodes, worker, origin, choice, delay):
    env = ReplayManagerEnv(episodes, worker, verify=True)
    env.reset(seed=0, options={'episode': 2, 'delay': delay})
    while env.frame < origin:
        _, _, done, _, _ = env.step(0)
        if done:
            raise ValueError('Baseline stopped before the trigger')
    env.choices = (choice,)
    while True:
        _, _, done, _, info = env.step(0)
        if done:
            break
    return {**info, 'through_frame': env.frame, 'crossed_388': env.frame > 388,
            'history': [row for row in info['history'] if row['frame'] >= origin]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin', type=int, default=320)
    parser.add_argument('--width', type=int, default=36)
    parser.add_argument('--event-aligned', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    episodes = load_capture(TEST, digest(BASE))
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits)
    delay = 1
    env = ReplayManagerEnv(episodes, worker, verify=True)
    env.reset(seed=0, options={'episode': 2, 'delay': delay})
    while env.frame < args.origin:
        _, _, done, _, _ = env.step(0)
        if done:
            raise ValueError('Baseline stopped before the trigger')
    if env.frame != args.origin:
        raise ValueError('Origin must be a decision frame')
    obs = rebase(env.episode.samples[env.frame].observation, env.position)
    allowed = feasible_actions(obs, HOLD)
    baseline = env.choices[0]
    closing, horizon_long = event_horizon(env.scene) if args.event_aligned else (None, LONG_HORIZON)
    rows = []
    for action in range(len(VECTORS)):
        row = {'action': action, 'allowed': action in allowed, 'forecasts': {}}
        rows.append(row)
        if action not in allowed:
            continue
        short_choices = {}
        for horizon in sorted({SHORT_HORIZON, horizon_long}):
            for modeled_delay in (0, 1):
                start = time.perf_counter()
                route, choice = seek(env.scene, env.belief, allowed, action, horizon, modeled_delay, args.width)
                row['forecasts'][f'{horizon}_{modeled_delay}'] = {
                    'actions': route.actions if route else None,
                    'leaf_ttc': route.ttc if route else None,
                    'prefix_heuristic_60': bool(choice and SHORT_HORIZON + choice.ttc >= LONG_HORIZON),
                    'ms': 1000 * (time.perf_counter() - start),
                }
                if choice:
                    short_choices[horizon, modeled_delay] = choice
        row['manager_options'] = [i for i, choice in enumerate(env.choices) if choice.route.actions[0] == action]
        if action == baseline.route.actions[0]:
            choice = baseline
        else:
            choice = short_choices.get((horizon_long, delay)) or short_choices.get((SHORT_HORIZON, delay))
            choice = choice or short_choices.get((SHORT_HORIZON, 0))
        if choice:
            row['closed_loop'] = branch(episodes, worker, args.origin, choice, delay)
        print(action, row['manager_options'],
              [key for key, value in row['forecasts'].items() if value['actions']],
              row.get('closed_loop', {}).get('through_frame'), flush=True)
    report = {'source_sha256': digest(TEST), 'origin': args.origin, 'delay': delay,
              'closure_frame': args.origin + closing if closing is not None else None,
              'long_horizon': horizon_long,
              'position': env.position.tolist(), 'danger': env.danger, 'width_per_first_action': args.width,
              'rule_action': baseline.route.actions[0], 'candidates': rows}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
