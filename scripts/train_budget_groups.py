"""Collect whole-trajectory shortlist misses and train separate survival preferences."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.budget_ranker import direction_features
from agent.replay_hrl import HOLD
from scripts.audit_replay_replanning_scores import forecast_candidates
from scripts.correct_budget_ranker import PREVIOUS, load_student, trajectory_states, verified_directions
from scripts.evaluate_replay_budget import MANAGERS, small_search
from scripts.train_budget_ranker import FULL, GRID, REFERENCE, SEED, fit, mask, offline, rollout, teacher_data
from scripts.train_replay_hrl import PPO, digest
from scripts.train_replay_mixed_windows import restore


STRIDE = 64
PROBES = 6


def make_row(view, obs, scores):
    allowed = feasible_actions(obs, HOLD)
    features = direction_features(view.scene, obs, view.belief, view.worker.logits(obs))
    values = np.full(9, -1.)
    for action, score in scores.items():
        values[action] = score
    return features, mask(allowed), mask(scores), max(scores, key=scores.get), values, view.frame


def short_set(view, obs):
    return [a for a in feasible_actions(obs, HOLD) if small_search(view.scene, view.belief, [a], 2)[0]]


def handoffs(anchor, case):
    env, _ = restore(anchor, duration=None)
    env.fixed_delay = 0
    for h in case['controller']['history']:
        if env.frame != h['frame'] or not np.array_equal(env.position, h['start']):
            raise AssertionError('Teacher handoff drift')
        if env.frame > anchor.frame and (env.frame - anchor.frame) % STRIDE == 0:
            yield env
        _, _, done, _, _ = env.step(h['option'])


def inspect_run(origin, run, short_rows, records, probe_states):
    states = list(trajectory_states(origin, run))
    for i, (view, obs) in enumerate(states):
        top = run['decisions'][i]['shortlist']
        history = run['history']
        executed = history[i]['request'] if i < len(history) else None
        if executed == top[0]:
            continue
        valid = short_set(view, obs)
        missed = not set(top).intersection(valid)
        if (executed is None) != missed:
            raise AssertionError('Shortlist replay mismatch')
        records.append(dict(start=run['start'], delay=run['delay'], frame=view.frame,
                            position=list(view.position), top=top, valid=valid, missed=missed))
        if valid:
            short_rows.append(make_row(view, obs, dict.fromkeys(valid, 1.)))
        # Probe before a stop, while alternatives may still have a continuation.
        j = max(0, i - 2)
        prev, prev_obs = states[j]
        probe_states.append((not missed, prev.frame, run['delay'], prev, prev_obs))


def harvest(anchor, case, base, manager):
    short_rows, long_rows, records, probes, forecasts, runs = [], [], [], [], [], []
    def run_at(origin, duration):
        for delay in (0, 1):
            run = rollout(origin, delay, base, duration)
            runs.append(dict(start=run['start'], end=run['end'], delay=delay,
                             decisions=len(run['decisions']), outcome=run['outcome']))
            inspect_run(origin, run, short_rows, records, probes)
    run_at(anchor, None)
    for origin in handoffs(anchor, case):
        run_at(origin, STRIDE)
    seen = []
    for _, frame, delay, view, obs in sorted(probes, key=lambda row: row[:3]):
        if any(abs(frame - f) < 16 for f in seen):
            continue
        offered = sorted(verified_directions(view, obs).intersection(short_set(view, obs)))
        if len(offered) < 2:
            continue
        seen.append(frame)
        branches = {a: [b['frames'] for b in r['branches']] for a, r in
                    forecast_candidates(view, offered, view.worker, 60, manager)}
        scores = {a: float(np.mean(b)) for a, b in branches.items()}
        distinct = len(set(scores.values())) > 1
        if distinct:
            long_rows.append(make_row(view, obs, scores))
        forecasts.append(dict(frame=frame, delay=delay, position=list(view.position),
                              branches=branches, scores=scores, distinct=distinct))
        if len(seen) == PROBES:
            break
    print('Harvest', anchor.episode.source, len(records), len(long_rows), flush=True)
    return short_rows, long_rows, dict(source=anchor.episode.source, runs=runs, short=records, long=forecasts)


def different_scores(row):
    values = row[4]
    return len(np.unique(values[values >= 0])) > 1


def group_metrics(rows, model):
    return offline(rows, model) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    parent = json.loads((PREVIOUS / 'report.json').read_text())
    sources, anchors, data, _ = teacher_data()
    sources.update({str(p): digest(p) for p in (REFERENCE, FULL, GRID)})
    if sources != parent['sources']:
        raise ValueError('Parent source mismatch')
    teacher = json.loads(REFERENCE.read_text())
    manager = PPO.load(MANAGERS['paired'], device='cpu')
    report = dict(seed=SEED, stride=STRIDE, max_probes=PROBES, sources=sources,
                  parent_sha256=digest(PREVIOUS / 'report.json'), folds=[])
    for previous in parent['folds']:
        held = previous['held']
        train, = previous['train_sources']
        if held == train or set(data) != {held, train}:
            raise AssertionError('Capture leakage')
        anchor, = [a for (s, _), a in anchors.items() if s == train]
        test, = [a for (s, _), a in anchors.items() if s == held]
        case, = [c for c in teacher['cases'] if c['controller']['source'] == train and
                 c['controller']['episode'] == anchor.episode.number and c['policy'] == 'old' and c['delay'] == 0]
        base = load_student(PREVIOUS, previous['checkpoint'], previous['sha256'], parent['pooled']['inputs'])
        short, fresh, harvest_info = harvest(anchor, case, base, manager)
        long = fresh + [row for row in data[train] if different_scores(row)]
        # Fixed held-out states from the original student's paths.
        missed_rows, recovered_rows = [], []
        for run in previous['rollouts']:
            if run['controller'] != 'student':
                continue
            rows, records = [], []
            inspect_run(test, run, rows, records, [])
            labeled = [r for r in records if r['valid']]
            for row, record in zip(rows, labeled, strict=True):
                (missed_rows if record['missed'] else recovered_rows).append(row)
        scored = [row for row in data[held] if different_scores(row)]
        fold = dict(held=held, train_sources=[train], harvest=harvest_info, short_labels=len(short),
                    fresh_long=len(fresh), archive_long=len(long)-len(fresh), variants={})
        for name in ('baseline', 'short', 'joint'):
            model = base
            weight_hash = previous['sha256']
            if name != 'baseline':
                prefs = long if name == 'joint' else []
                original = data[train]
                model, _ = fit(original + short + prefs,
                               [1.]*len(original) + [4.]*len(short) + [4.]*len(prefs), len(prefs))
                path = out / (name + '-' + previous['checkpoint'])
                torch.save(model.state_dict(), path)
                weight_hash = digest(path)
                model = load_student(out, path.name, weight_hash, parent['pooled']['inputs'])
            runs = [rollout(test, delay, model) for delay in (0, 1)]
            if any(d['checked'] > 20 or len(d['shortlist']) > 2 for r in runs for d in r['decisions']):
                raise AssertionError('Budget exceeded')
            if name == 'baseline':
                original_runs = [r for r in previous['rollouts'] if r['controller'] == 'student']
                if any((a['history'],a['end']) != (b['history'],b['end']) for a,b in zip(runs, original_runs, strict=True)):
                    raise AssertionError('Parent replay changed')
            fold['variants'][name] = dict(sha256=weight_hash, teacher=offline(data[held], model),
                                          missed=group_metrics(missed_rows, model),
                                          recovered=group_metrics(recovered_rows, model),
                                          distinct_long=group_metrics(scored, model), rollouts=runs)
            print(held, name, [r['end'] for r in runs], flush=True)
        report['folds'].append(fold)
        (out / 'progress.json').write_text(json.dumps(report, allow_nan=False))
    (out / 'report.json').write_text(json.dumps(report, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
