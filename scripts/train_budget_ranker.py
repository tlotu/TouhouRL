"""Distill the full replay teacher and cross-validate by original capture."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from agent.boundary_control import feasible_actions
from agent.budget_ranker import DirectionRanker, direction_features
from agent.replay_hrl import HOLD, ReplayWorker, rebase
from scripts.diagnose_replay_mixed_windows import predict
from scripts.evaluate_replay_budget import MANAGERS, REFERENCE, small_search
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, attack_position_summary, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


FULL = Path('runs/hrl-young-auto-reselection-002.json')
GRID = Path('runs/hrl-young-budget-grid-001.json')
SEED = 271


def key(case):
    source = case['controller']
    return source['source'], source['episode'], case['policy'], case['delay']


def mask(actions):
    return np.isin(np.arange(9), list(actions))


def collect(anchor, case, events):
    env, _ = restore(anchor, duration=None)
    env.fixed_delay = case['delay']
    rows = []
    for movement in case['controller']['history']:
        if env.frame != movement['frame'] or not np.array_equal(env.position, movement['start']):
            raise AssertionError('Teacher replay drift')
        obs = rebase(env.episode.samples[env.frame].observation, env.position)
        prior = env.worker.logits(obs)
        offered = mask(c.route.actions[0] for c in env.choices)
        target = movement['request']
        known = np.full(9, -1.)
        if env.frame in events:
            for action, value in events[env.frame]['mean_frames'].items():
                known[int(action)] = value
            if not (known >= 0).any() and offered.sum() != 1:
                raise AssertionError('Unscored event with multiple candidates')
            if (known >= 0).any() and (not np.array_equal(known >= 0, offered) or known[target] != known.max()):
                raise AssertionError('Incomplete or inconsistent teacher event')
        rows.append((direction_features(env.scene, obs, env.belief, prior), mask(feasible_actions(obs, HOLD)),
                     offered, target, known, env.frame))
        slot = movement['option']
        if env.choices[slot].route.actions[0] != target:
            raise AssertionError('Teacher choice changed')
        _, _, done, _, _ = env.step(slot)
    if not done or env.frame != env.episode.end:
        raise AssertionError('Teacher did not reach trace end')
    return rows


def fit(rows, emphasis=None, preference_only=0):
    x, legal, offered, target, known, _ = map(np.asarray, zip(*rows))
    # Repeated visits from manager and delay variants share one unit of weight.
    keys = [v.tobytes() for v in x]
    counts = Counter(keys)
    weight = torch.tensor([1. / counts[k] for k in keys], dtype=torch.float32)
    if emphasis is not None:
        weight *= torch.as_tensor(emphasis, dtype=weight.dtype)
    x, legal, offered, target, known = map(torch.as_tensor, (x, legal, offered, target, known))
    use_feasibility = torch.arange(len(rows)) < len(rows) - preference_only
    event = known.amax(1) >= 0
    good = F.one_hot(target, 9).bool()
    good[event] = known[event] == known[event].amax(1, keepdim=True)
    torch.manual_seed(SEED)
    model = DirectionRanker(x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    order = torch.Generator().manual_seed(SEED)
    for _ in range(60):
        for idx in torch.randperm(len(x), generator=order).split(96):
            logits = model(x[idx])
            # Prefer any top-scoring verified direction at a scored event.
            # Else imitate the demonstrated choice among verified directions.
            preference = logits.masked_fill(~offered[idx], -1e6).logsumexp(1) - logits.masked_fill(
                ~good[idx], -1e6).logsumexp(1)
            feasible = F.binary_cross_entropy_with_logits(logits, offered[idx].float(), reduction='none')
            feasibility = (feasible * legal[idx]).sum(1) / legal[idx].sum(1)
            loss = ((preference + feasibility * use_feasibility[idx]) * weight[idx]).sum() / weight[idx].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), len(counts)


class StudentEnv(CoverageEnv):
    def _plan_options(self, obs):
        start = perf_counter()
        prior = self.worker.logits(obs)
        allowed = feasible_actions(obs, HOLD)
        scores = prior
        if self.model is not None:
            features = direction_features(self.scene, obs, self.belief, prior)
            with torch.inference_mode():
                scores = self.model(torch.from_numpy(features)).numpy()
        top = sorted(allowed, key=lambda action: (-scores[action], action))[:2]
        choices, checked = small_search(self.scene, self.belief, top, 2)
        ms = (perf_counter() - start) * 1000
        self.audit.append(dict(frame=self.frame, shortlist=top, checked=checked, ms=ms))
        return choices, checked, ms


def rollout(anchor, delay, model, duration=None):
    env, _ = restore(anchor, duration=duration)
    env.__class__ = StudentEnv
    env.fixed_delay, env.model, env.audit = delay, model, []
    env._prepare()
    while True:
        _, _, done, _, info = env.step(0)
        if done:
            break
    info.update(source=env.episode.source, episode=env.episode.number)
    return {**{k: info[k] for k in ('source', 'episode', 'history', 'outcome', 'first_below_padding_frame',
                                     'min_clearance_px')}, 'attack': attack_position_summary(env.episodes, [info]),
            'delay': delay, 'start': anchor.frame, 'end': env.frame, 'decisions': env.audit}


def offline(rows, model):
    x, legal, offered, label, known, _ = map(np.asarray, zip(*rows))
    # The prior feature is clipped for the network. Use the original ordinary
    # shortlist from the matching budget run for baseline metrics outside here.
    with torch.inference_mode():
        scores = model(torch.from_numpy(x)).numpy()
    return metrics(np.argsort(-np.where(legal, scores, -np.inf), kind='stable')[:, :2], offered, label, known)


def metrics(top, offered, label, known):
    event = known.max(1) >= 0
    good = np.eye(9, dtype=bool)[label]
    good[event] = known[event] == known[event].max(axis=1, keepdims=True)
    hits = np.take_along_axis(good, top, axis=1)
    unique = offered.sum(axis=1) == 1
    return dict(n=len(label), top1=int(hits[:, 0].sum()), top2=int(hits.any(1).sum()),
                teacher_top1=int((top[:, 0] == label).sum()), teacher_top2=int((top == label[:, None]).any(1).sum()),
                events=int(event.sum()), event_top2=int(hits[event].any(1).sum()),
                unique_verified=int(unique.sum()), unique_top2=int(hits[unique].any(1).sum()))


def teacher_data():
    ref, full, grid = (json.loads(p.read_text()) for p in (REFERENCE, FULL, GRID))
    sources = ref['sources']
    if sources != full['sources'] or sources != grid['sources'] or any(digest(Path(k)) != v for k,v in sources.items()):
        raise ValueError('Source mismatch')
    full_cases = {key(c): c for c in full['cases']}
    budgets = {(c['source'], c['episode'], c['policy'], c['delay']): c for c in grid['cases']
               if c['width'] == c['depth'] == 2 and c['teacher']}
    episodes = [e for p in TRAIN for e in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(MANAGERS['old'], device='cpu')
    anchors, data, baseline_top = {}, {}, {}
    for c in ref['cases']:
        source, ep, _, _ = k = key(c)
        if (source, ep) not in anchors:
            anchor = CoverageEnv(episodes, worker, verify=True)
            index = next(i for i,e in enumerate(episodes) if (e.source, e.number) == (source, ep))
            obs, _ = anchor.reset(seed=0, options={'episode': index, 'delay': 0})
            while anchor.frame < c['anchor']:
                obs, _, done, _, _ = anchor.step(predict(old, obs))
                if done:
                    raise AssertionError('Anchor unreachable')
            anchors[source, ep] = anchor
        original = full_cases[k]['controller']
        if original['history'] != c['controller']['history']:
            raise AssertionError('Full teacher trajectory differs')
        rows = collect(anchors[source, ep], c, {e['frame']: e for e in original['events']})
        ordinary = budgets[k]['decisions']
        if [(v[5], v[3]) for v in rows] != [(v['frame'], v['teacher']) for v in ordinary]:
            raise AssertionError('Reference decision mismatch')
        data.setdefault(source, []).extend(rows)
        baseline_top.setdefault(source, []).extend(v['ordinary'] for v in ordinary)
        print('Collected', k, len(rows), flush=True)
    return sources, anchors, data, baseline_top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    sources, anchors, data, baseline_top = teacher_data()
    report = dict(seed=SEED, sources={**sources, **{str(p): digest(p) for p in (REFERENCE, FULL, GRID)}}, folds=[])
    for source, held in data.items():
        train = [r for other, rows in data.items() if other != source for r in rows]
        model, unique = fit(train)
        stem = 'holdout-' + Path(source).stem
        checkpoint = args.output / (stem + '.pt')
        torch.save(model.state_dict(), checkpoint)
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        _, _, offered, label, known, _ = map(np.asarray, zip(*held))
        fold = dict(held=source, train_sources=[s for s in data if s != source], train_rows=len(train),
                    train_unique=unique, checkpoint=checkpoint.name, sha256=digest(checkpoint),
                    student=offline(held, model), ordinary=metrics(np.asarray(baseline_top[source]), offered, label, known),
                    rollouts=[])
        for (capture, ep), anchor in anchors.items():
            if capture == source:
                for delay in (0, 1):
                    for name, scorer in [('ordinary', None), ('student', model)]:
                        result = rollout(anchor, delay, scorer)
                        fold['rollouts'].append(dict(controller=name, **result))
                        print(stem, delay, name, result['outcome'], result['end'], flush=True)
        report['folds'].append(fold)
    pooled = [r for rows in data.values() for r in rows]
    model, unique = fit(pooled)
    checkpoint = args.output / 'student-all.pt'
    torch.save(model.state_dict(), checkpoint)
    report['pooled'] = dict(checkpoint=checkpoint.name, sha256=digest(checkpoint), train_rows=len(pooled),
                            unique=unique, inputs=pooled[0][0].shape[-1])
    (args.output / 'report.json').write_text(json.dumps(report, allow_nan=False, indent=2))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()



