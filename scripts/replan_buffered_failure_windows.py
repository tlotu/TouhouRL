"""Check four-frame feedback with a sixteen-frame robust lookahead."""
import argparse
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from agent.boundary_control import feasible_actions
from agent.direction_buffer import DirectionBuffer
from agent.forward_model import ForwardModel, scene_from_observation
from agent.short_planner import HOLD, SHORT_HORIZON, VECTORS
from agent.swept_collision import SweptCollision
from scripts.analyze_input_switch import TOLERANCE
from scripts.audit_buffered_intervention import PADDING, frame_start, motions, path_points, safe_paths, transitions
from scripts.replay_buffered_closed_loop import PLAYER_SCALE, digest, player_position


def key_position(point):
    return tuple(round(float(value), 6) for value in point)


def calibrate(rows, scenes):
    """Filter hidden queue states by the observed endpoint, one sample at a time."""
    buffers = frozenset({DirectionBuffer()})
    diagnostics, priors = [], {}
    for row, scene in zip(rows, scenes):
        frame = frame_start(row)
        if not buffers or row['terminated'] or row['truncated']:
            break
        priors[frame] = buffers
        end = player_position({'player': np.asarray(row['next_observation']['player'], dtype=np.float32)})
        xmin, xmax, ymin, ymax = scene.bounds
        matches, residuals = [], []
        for buffer in buffers:
            for delay in (0, 1):
                moves, tail = buffer.advance(row['action'], row['info']['delta_frames'], delay)
                point = np.asarray(scene.player)
                for direction in moves:
                    point = np.clip(point + scene.speed * VECTORS[direction], (xmin, ymin), (xmax, ymax))
                error = float(np.linalg.norm(point - end))
                residuals.append(error)
                if error <= TOLERANCE:
                    matches.append((delay, moves, tail))
        buffers = frozenset(tail for _, _, tail in matches)
        diagnostics.append({'frame': frame, 'action': row['action'],
                            'observed_displacement_px': (end - scene.player).tolist(),
                            'min_error_px': min(residuals), 'matched_delays': sorted({d for d, _, _ in matches}),
                            'matched_motion': sorted({m for _, m, _ in matches}),
                            'remaining_queue_states': len(buffers)})
    return priors, diagnostics


def allowed_at(position):
    return feasible_actions({'player': np.asarray(position) / PLAYER_SCALE - 1.}, HOLD)


class FeedbackPlanner:
    """Find a safe action for each distinguishable endpoint and queue belief."""

    def __init__(self, scene):
        padded = replace(scene, player_radius=scene.player_radius + PADDING)
        self.model = SweptCollision(padded)
        self.reference = ForwardModel(padded)
        self.successors = lru_cache(None)(self._successors)
        self.plan = lru_cache(None)(self._plan)

    def _successors(self, frame, position, buffers, action):
        if action not in allowed_at(position):
            return None
        options = sorted(motions(buffers, (action,)), key=repr)
        points = np.asarray(position) + self.model.scene.speed * VECTORS[
            np.asarray([moves for moves, _ in options])].cumsum(axis=1)
        starts = np.broadcast_to(position, (len(points), 2))
        clearance = self.model.clearance_cap
        for offset, ends in enumerate(points.transpose(1, 0, 2)):
            valid, gaps = self.model.edges(starts, ends, frame + offset, 1)
            if not valid.all():
                return None
            clearance = min(clearance, float(gaps.min()))
            starts = ends
        endpoints = defaultdict(set)
        for point, (_, buffer) in zip(points[:, -1], options):
            endpoints[key_position(point)].add(buffer)
        return clearance, {point: frozenset(tails) for point, tails in endpoints.items()}

    def _plan(self, frame, position, buffers, remaining, first=None):
        candidates = []
        for action in allowed_at(position) if first is None else (first,):
            result = self.successors(frame, position, buffers, action)
            if result is not None:
                clearance, endpoints = result
                candidates.append((-clearance, action, endpoints))
        for clearance, action, endpoints in sorted(candidates):
            children = {}
            if remaining > HOLD:
                for point, tails in endpoints.items():
                    child = self.plan(frame + HOLD, point, tails, remaining - HOLD)
                    if child is None:
                        break
                    children[point] = child
                if len(children) != len(endpoints):
                    continue
            return action, -clearance, endpoints, children
        return None

    def verify(self, node, frame, position, buffers):
        action, _, endpoints, children = node
        options = list(motions(buffers, (action,)))
        points = np.asarray(position) + self.model.scene.speed * VECTORS[
            np.asarray([moves for moves, _ in options])].cumsum(axis=1)
        starts = np.broadcast_to(position, (len(points), 2))
        for offset, ends in enumerate(points.transpose(1, 0, 2)):
            if not self.reference.edges(starts, ends, frame + offset, 1)[0].all():
                raise ValueError('Feedback route failed independent collision verification')
            starts = ends
        for point, child in children.items():
            self.verify(child, frame + HOLD, point, endpoints[point])

    def close(self):
        self.plan.cache_clear()
        self.successors.cache_clear()


def refreshed_replay(by_frame, origin, first_hold):
    """Replan from each possible observed endpoint using refreshed logged hazards."""
    frontier = first_hold[1] if first_hold is not None else {}
    rounds = []
    for elapsed in range(HOLD, SHORT_HORIZON, HOLD):
        if not frontier:
            break
        frame = origin + elapsed
        if frame not in by_frame:
            break
        planner = FeedbackPlanner(by_frame[frame])
        following = defaultdict(set)
        states = []
        for point, tails in sorted(frontier.items()):
            belief = frozenset(tails)
            node = planner.plan(0, point, belief, SHORT_HORIZON)
            state = {'position': point, 'queue_states': len(belief), 'action': None, 'endpoints': []}
            if node is not None:
                planner.verify(node, 0, point, belief)
                action, clearance, endpoints, _ = node
                state.update(action=action, first_hold_padded_clearance_px=clearance,
                             endpoints=sorted(endpoints))
                for endpoint, next_buffers in endpoints.items():
                    following[endpoint].update(next_buffers)
            else:
                state['longest_safe_lookahead_frames'] = max(
                    (length for length in range(HOLD, SHORT_HORIZON, HOLD)
                     if planner.plan(0, point, belief, length) is not None), default=0)
            states.append(state)
        planner.close()
        rounds.append({'frame': frame, 'states': states})
        frontier = following
    return {'rounds': rounds,
            'all_branches_kept': len(rounds) == SHORT_HORIZON // HOLD - 1
            and all(state['action'] is not None for step in rounds for state in step['states'])}


def audit(trace, episode, frames):
    for _, rows, _ in transitions(trace):
        if rows[0]['episode'] + 1 == episode:
            break
    else:
        raise ValueError('Episode missing from trace')
    for previous, current in zip(rows, rows[1:]):
        if (previous['info']['elapsed_frames'] != frame_start(current)
                or previous['next_observation'] != current['observation']):
            raise ValueError('Trace observations are not aligned')
    scenes = [scene_from_observation(row['observation'])[0] for row in rows]
    by_frame = {frame_start(row): scene for row, scene in zip(rows, scenes)}
    priors, calibration = calibrate(rows, scenes)
    result = []
    for row in rows:
        frame = frame_start(row)
        if frame not in frames:
            continue
        scene = by_frame[frame]
        point = key_position(scene.player)
        if frame not in priors:
            raise ValueError(f'Cannot calibrate the input queue at frame {frame}')
        buffers = priors[frame]
        shadow = row['planner_shadow']
        actions = tuple(shadow['route_actions'][:SHORT_HORIZON // HOLD])
        if row['commitment_active'] or len(actions) * HOLD != SHORT_HORIZON or actions[0] == row['action']:
            raise ValueError('Expected a fresh alternative with a sixteen-frame prefix')
        planner = FeedbackPlanner(scene)
        fixed = list(motions(buffers, actions))
        checks = safe_paths(planner.reference, path_points(planner.reference, fixed))
        record = {'frame': frame, 'position': point, 'initial_queue_states': len(buffers),
                  'logged_route': actions, 'fixed_safe_paths': int(checks.sum()), 'fixed_paths': len(fixed)}
        for label, first in (('baseline', row['action']), ('alternative', actions[0])):
            node = planner.plan(0, point, buffers, SHORT_HORIZON, first)
            first_hold = planner.successors(0, point, buffers, first)
            first_outcomes = []
            if first_hold is not None:
                planner.verify((first, *first_hold, {}), 0, point, buffers)
                for buffer in sorted(buffers, key=repr):
                    for delay in (0, 1):
                        moves, _ = buffer.advance(first, HOLD, delay)
                        end = np.asarray(point) + scene.speed * VECTORS[list(moves)].sum(axis=0)
                        first_outcomes.append({'delay': delay, 'motion': moves, 'position': key_position(end)})
            record[label] = {'first_action': first, 'first_hold_safe': first_hold is not None,
                             'first_hold_outcomes': first_outcomes,
                             'initial_lookahead_safe': node is not None,
                             'initial_longest_safe_lookahead_frames': max(
                                 (length for length in range(HOLD, SHORT_HORIZON + 1, HOLD)
                                  if planner.plan(0, point, buffers, length, first) is not None), default=0),
                             **refreshed_replay(by_frame, frame, first_hold)}
        planner.close()
        result.append(record)
    if len(result) != len(frames):
        raise ValueError('Requested frame missing from trace')
    return {'source': str(trace), 'source_sha256': digest(trace), 'episode': episode,
            'recorded_terminal_frame': rows[-1]['info']['elapsed_frames'],
            'hold_frames': HOLD, 'lookahead_frames': SHORT_HORIZON, 'padding_px': PADDING,
            'sample_position_tolerance_px': TOLERANCE, 'calibration': calibration, 'windows': result,
            'limitations': [
                'Only sampled endpoints constrain the input queue. The calibration stops at the first mismatch.',
                'Feedback observes modeled endpoints. Hidden queue states sharing an endpoint must share a decision.',
                'Replanning reuses hazards sampled along the recorded trajectory. Future spawns and player-dependent hazards are not simulated.',
                'Lookahead may extend beyond the final recorded frame using the last available forecast.',
                'The finite-horizon search is offline and its success does not imply real-time feasibility.',
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--episode', type=int, required=True, help='One-based episode number')
    parser.add_argument('--frame', type=int, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.trace, args.episode, set(args.frame))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for window in result['windows']:
        print(window['frame'], {name: (window[name]['initial_lookahead_safe'], window[name]['all_branches_kept'])
                               for name in ('baseline', 'alternative')})


if __name__ == '__main__':
    main()
