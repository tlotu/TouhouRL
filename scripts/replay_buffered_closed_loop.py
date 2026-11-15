"""Compare two first holds, then rerun the frozen PPO on each simulated branch."""
import argparse
from collections import Counter
from functools import cache
from hashlib import file_digest
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from agent.boundary_control import feasible_actions
from agent.direction_buffer import DirectionBuffer
from agent.forward_model import ForwardModel, scene_from_observation
from agent.laser_evaluation import LaserEvaluationCommitment
from agent.laser_policy import LaserRiskPolicy
from agent.ppo_backend import PPOBackend
from agent.short_planner import HOLD, VECTORS
from scripts.audit_buffered_intervention import DEADLINE, PADDING, compare_contacts, frame_start, transitions


PLAYER_SCALE = np.asarray((192., 224.))
RELATIVE_SCALE = 2 * PLAYER_SCALE


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def player_position(obs):
    return (obs['player'][:2].astype(float) + 1.) * PLAYER_SCALE


def rebase(obs, position):
    """Move the player and recompute relative rows for the objects we observed."""
    displacement = np.asarray(position) - player_position(obs)
    result = dict(obs)
    player = obs['player'].copy()
    player[:2] = np.clip(np.asarray(position) / PLAYER_SCALE - 1., -1., 1.)
    result['player'] = player
    for key in ('bullets', 'enemies'):
        rows = obs[key].copy()
        present = rows[:, -1] > 0
        rows[present, :2] = np.clip(rows[present, :2] - displacement / RELATIVE_SCALE, -1., 1.)
        result[key] = rows
    return result


def verify_logged_policy(rows, observations, logits, slack):
    """Separate checkpoint/observation identity from simulated movement error."""
    frame = 0
    controller = LaserEvaluationCommitment(logits, lambda: frame, candidate_slack=slack)
    largest_error = 0.
    for row, obs in zip(rows, observations):
        frame = frame_start(row)
        action = controller(obs)
        last = controller.annotate({'event': 'transition', 'action': action})
        if action != row['action'] or last['commitment_active'] != row['commitment_active']:
            raise ValueError('Frozen checkpoint did not reproduce the logged controller')
        if not row['commitment_active']:
            error = float(np.max(np.abs(np.asarray(last['actor_logits']) - row['actor_logits'])))
            largest_error = max(largest_error, error)
            if error > 1e-5 or last['laser_risk_context'] != row['laser_risk_context']:
                raise ValueError('Frozen checkpoint logits or risk gate differ from the trace')
    return largest_error


def rollout(rows, observations, models, start, first, buffer, delay, horizon, logits, slack):
    origin = frame = frame_start(rows[start])
    position = player_position(observations[start])
    controller = LaserEvaluationCommitment(logits, lambda: frame, candidate_slack=slack)
    controller.since, controller.held_action = origin, first
    contacts = [None, None]
    agreement = decisions = omitted = 0
    errors = []
    history = []
    elapsed = 0
    for index in range(start, len(rows)):
        row, obs = rows[index], observations[index]
        frame = frame_start(row)
        duration = min(row['info']['delta_frames'], horizon - elapsed)
        if frame != origin + elapsed:
            raise ValueError('Non-contiguous source sample')
        forced = elapsed < HOLD
        if forced:
            if elapsed + duration > HOLD:
                raise ValueError('Source sample crosses the first hold boundary')
            action = first
        else:
            action = controller(rebase(obs, position))
            controller.annotate({'event': 'transition', 'action': action})
        errors.append(float(np.linalg.norm(position - player_position(obs))))
        decisions += 1
        agreement += action == row['action']
        omitted = max(omitted, rows[index - 1]['info']['dropped_bullets'])
        history.append({'frame': frame, 'position': position.tolist(), 'action': action,
                        'recorded_action': row['action'], 'forced': forced})
        model = models(index)
        xmin, xmax, ymin, ymax = model.scene.bounds
        motion, buffer = buffer.advance(action, duration, delay)
        for offset, move in enumerate(motion):
            end = np.clip(position + model.scene.speed * VECTORS[move], (xmin, ymin), (xmax, ymax))
            _, gap = model.edges([position], [end], offset, 1)
            for i, margin in enumerate((0., PADDING)):
                if contacts[i] is None and gap[0] <= margin:
                    contacts[i] = elapsed + 1
            position = end
            elapsed += 1
        if elapsed == horizon:
            break
    return {'through_frame': origin + elapsed, 'first_contact_offset_frames': contacts,
            'max_dropped_bullets': omitted, 'recorded_request_matches': agreement, 'requests': decisions,
            'recorded_position_rmse_px': math.sqrt(sum(error ** 2 for error in errors) / len(errors)),
            'recorded_position_max_error_px': max(errors), 'trajectory': history}


def run_trace(source, horizon, logits, checkpoint_sha256):
    path = Path(source['source'])
    if digest(path) != source['sha256'] or source['model_sha256'] != checkpoint_sha256:
        raise ValueError('Replay source or checkpoint hash differs from the audit')
    pending = {(case['episode'], case['frame']): case for case in source['candidates']}
    cases = []
    verified_rows = 0
    logit_error = 0.
    for header, rows, _ in transitions(path):
        if header['model_sha256'] != checkpoint_sha256 or header['hold_frames'] != HOLD:
            raise ValueError('Unexpected checkpoint or hold configuration')
        slack = header['candidate_slack']
        observations = [{key: np.asarray(value, dtype=np.float32) for key, value in row['observation'].items()}
                        for row in rows]
        logit_error = max(logit_error, verify_logged_policy(rows, observations, logits, slack))
        verified_rows += len(rows)
        buffers = [DirectionBuffer(), DirectionBuffer()]

        @cache
        def models(index):
            return ForwardModel(scene_from_observation(observations[index])[0])

        for index, row in enumerate(rows):
            frame = frame_start(row)
            key = row['episode'] + 1, frame
            if key in pending:
                case = pending.pop(key)
                firsts = {'baseline_first': case['baseline'], 'alternative_first': case['proposal']}
                if (index == 0 or row['commitment_active'] or firsts['baseline_first'] != row['action']
                        or firsts['alternative_first'] != row['planner_shadow']['suggested_action']
                        or any(action not in feasible_actions(observations[index], HOLD) for action in firsts.values())):
                    raise ValueError('Audit case differs from the recorded decision')
                replay = {name: [rollout(rows, observations, models, index, first, buffers[delay],
                                         delay, horizon, logits, slack) for delay in (0, 1)]
                          for name, first in firsts.items()}
                if any(a['through_frame'] != b['through_frame']
                       for a, b in zip(replay['baseline_first'], replay['alternative_first'])):
                    raise ValueError('Counterfactual windows must end at the same frame')
                paired = compare_contacts(replay, frame, baseline='baseline_first')['alternative_first']
                cases.append({**{k: case[k] for k in ('episode', 'frame', 'reason', 'episode_outcome',
                                                      'projected_with_guard_ms')},
                              'first_actions': firsts, 'branches': replay, 'contact_comparison': paired})
            for delay in (0, 1):
                _, buffers[delay] = buffers[delay].advance(row['action'], row['info']['delta_frames'], delay)
    if pending:
        raise ValueError('Could not find every audited decision')
    return {'source': source['source'], 'sha256': source['sha256'], 'verified_rows': verified_rows,
            'max_logged_logit_error': logit_error, 'cases': cases}


def summarize(cases):
    return {margin: dict(Counter(case['contact_comparison'][margin]['classification'] for case in cases))
            for margin in ('nominal', 'padded')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path, help='Frozen model.zip')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    horizon = audit['counterfactual_horizon_frames']
    sha256 = digest(args.checkpoint)
    if any(trace['model_sha256'] != sha256 for trace in audit['traces']):
        raise ValueError('Checkpoint hash differs from the audit')
    torch.set_num_threads(1)
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(str(args.checkpoint), device='cpu')
    if type(backend.model.policy) is not LaserRiskPolicy:
        raise ValueError('Expected the frozen laser PPO checkpoint')
    traces = [run_trace(trace, horizon, backend.action_logits, sha256) for trace in audit['traces']]
    cases = [case for trace in traces for case in trace['cases']]
    within_budget = [case for case in cases if case['projected_with_guard_ms'] <= 1000 * DEADLINE]
    report = {'audit_sha256': digest(args.audit), 'model_sha256': sha256, 'horizon_frames': horizon,
              'start_delays': [0, 1], 'hold_frames': HOLD, 'traces': traces,
              'contact_summary': {'all': summarize(cases), 'within_audit_budget': summarize(within_budget)},
              'limitations': [
                  'Both branches use recorded hazard snapshots; no enemy or bullet spawns respond to the new path.',
                  'Rebasing uses recorded float32 rows. Missing bullets and clipped coordinates cannot be recovered.',
                  'First contacts describe this model. Branches continue after contact for paired replay diagnostics.',
                  'Budget eligibility comes from the previous offline audit, not live timings.',
              ]}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['contact_summary'], indent=2))


if __name__ == '__main__':
    main()
