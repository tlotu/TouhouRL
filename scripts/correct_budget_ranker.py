"""One on-policy correction round for the fixed two-direction replay student."""
import argparse
from copy import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.budget_ranker import DirectionRanker, direction_features
from agent.replay_hrl import HOLD, hold_outcomes, rebase
from scripts.audit_replay_replanning_scores import forecast_candidates
from scripts.evaluate_replay_budget import MANAGERS, small_search
from scripts.train_budget_ranker import FULL, GRID, REFERENCE, SEED, fit, mask, offline, rollout, teacher_data
from scripts.train_replay_hrl import PPO, digest


PREVIOUS = Path('runs/hrl-young-budget-student-002')
TAIL = 5
EMPHASIS = 8.


def trajectory_states(anchor, run):
    belief, position = anchor.belief, anchor.position
    for i, decision in enumerate(run['decisions']):
        frame = decision['frame']
        sample = anchor.episode.samples[frame]
        view = copy(anchor)
        view.frame, view.position, view.belief = frame, np.asarray(position), belief
        view.scene = replace(sample.scene, player=tuple(position))
        yield view, rebase(sample.observation, position)
        if i < len(run['history']):
            movement = run['history'][i]
            if movement['frame'] != frame or not np.array_equal(position, movement['start']):
                raise AssertionError('Correction trajectory mismatch')
            outcomes = hold_outcomes(view.scene, belief, movement['request'])
            position = movement['end']
            belief = frozenset(outcomes[tuple(np.round(position, 6))])


def verified_directions(view, obs):
    choices, _, _ = view.worker.options(view.scene, obs, view.belief, waypoint_coverage=True)
    return {c.route.actions[0] for c in choices}


def corrections(anchor, model, manager, terminal_windows):
    rows, records, trajectories = [], [], []
    for delay in (0, 1):
        run = rollout(anchor, delay, model)
        trajectories.append(run)
        states = list(trajectory_states(anchor, run))
        end = len(states) - 1
        if run['outcome'] == 'trace_end' and not terminal_windows:
            # Find an actionable bottleneck on the training trajectory.
            # Prefer a missed proposal set, then a narrow teacher option set.
            priorities = []
            for i, (view, obs) in enumerate(states):
                verified = verified_directions(view, obs)
                if verified:
                    shortlist = run['decisions'][i]['shortlist']
                    priorities.append((bool(verified.intersection(shortlist)), len(verified),
                                       shortlist[0] in verified, i))
            if priorities:
                end = min(priorities)[-1]
        for i in range(max(0, end - TAIL + 1), end + 1):
            view, obs = states[i]
            allowed = feasible_actions(obs, HOLD)
            short = [a for a in allowed if small_search(view.scene, view.belief, [a], 2)[0]]
            wide = verified_directions(view, obs)
            offered = sorted(wide.intersection(short))
            scores, branches = {}, {}
            for a, result in forecast_candidates(view, offered, anchor.worker, 60, manager):
                durations = [b['frames'] for b in result['branches']]
                scores[a] = float(np.mean(durations))
                branches[a] = durations
            records.append(dict(source=anchor.episode.source, delay=delay, frame=view.frame, position=list(view.position),
                                short=short, wide=sorted(wide), scores=scores, branches=branches,
                                shortlist=run['decisions'][i]['shortlist']))
            if scores:
                known = np.full(9, -1.)
                for a, score in scores.items():
                    known[a] = score
                rows.append((direction_features(view.scene, obs, view.belief, anchor.worker.logits(obs)),
                             mask(allowed), mask(offered), max(scores, key=scores.get), known, view.frame))
        print('Labeled', anchor.episode.source, delay, run['end'], 'window end', states[end][0].frame, flush=True)
    return rows, records, trajectories


def load_student(folder, filename, expected, inputs):
    path = folder / filename
    if digest(path) != expected:
        raise ValueError('Student checkpoint changed')
    model = DirectionRanker(inputs)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--terminal-windows', action='store_true',
                        help='Use the final window even when the training trace reaches its end.')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    parent = json.loads((PREVIOUS / 'report.json').read_text())
    sources, anchors, data, _ = teacher_data()
    sources.update({str(p): digest(p) for p in (REFERENCE, FULL, GRID)})
    if sources != parent['sources']:
        raise ValueError('Parent training sources changed')
    manager = PPO.load(MANAGERS['paired'], device='cpu')
    report = dict(seed=SEED, sources=sources, parent_sha256=digest(PREVIOUS / 'report.json'),
                  emphasis=EMPHASIS, tail=TAIL, forecast=60, manager='paired', terminal_windows=args.terminal_windows, folds=[])
    extra = []
    for previous in parent['folds']:
        held = previous['held']
        train_source, = previous['train_sources']
        if held == train_source or set(data) != {held, train_source}:
            raise AssertionError('Capture split mismatch')
        train_anchor, = [a for (s, _), a in anchors.items() if s == train_source]
        test_anchor, = [a for (s, _), a in anchors.items() if s == held]
        base = load_student(PREVIOUS, previous['checkpoint'], previous['sha256'], parent['pooled']['inputs'])
        added, labels, train_runs = corrections(train_anchor, base, manager, args.terminal_windows)
        train = data[train_source]
        model, unique = fit(train + added, [1.] * len(train) + [EMPHASIS] * len(added))
        path = args.output / previous['checkpoint']
        torch.save(model.state_dict(), path)
        model = load_student(args.output, path.name, digest(path), parent['pooled']['inputs'])
        fold = dict(held=held, train_sources=[train_source], original_rows=len(train), correction_rows=len(added),
                    unique=unique, labels=labels, training_trajectories=train_runs, student=offline(data[held], model),
                    correction_before=offline(added, base) if added else None,
                    correction_after=offline(added, model) if added else None,
                    previous=previous['student'], checkpoint=path.name, sha256=digest(path), rollouts=[])
        for delay in (0, 1):
            result = rollout(test_anchor, delay, model)
            if any(d['checked'] > 20 or len(d['shortlist']) > 2 for d in result['decisions']):
                raise AssertionError('Student search exceeded budget')
            fold['rollouts'].append(dict(controller='student', **result))
            print('Held', held, delay, result['outcome'], result['end'], flush=True)
        report['folds'].append(fold)
        extra.extend(added)
        (args.output / 'progress.json').write_text(json.dumps(report, allow_nan=False))
    train = [r for rows in data.values() for r in rows]
    model, unique = fit(train + extra, [1.] * len(train) + [EMPHASIS] * len(extra))
    path = args.output / 'student-all.pt'
    torch.save(model.state_dict(), path)
    load_student(args.output, path.name, digest(path), parent['pooled']['inputs'])
    report['pooled'] = dict(checkpoint=path.name, sha256=digest(path), unique=unique, corrections=len(extra))
    (args.output / 'report.json').write_text(json.dumps(report, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()



