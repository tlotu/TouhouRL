"""Measure short-route loss and test one-hold escapes before replay stops."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent.boundary_control import feasible_actions
from agent.replay_hrl import HOLD, PADDING, SHORT_HORIZON, Choice, ReplayWorker, rebase
from agent.short_planner import Route, VECTORS
from scripts.audit_replay_terminal_routes import feasible_prefixes
from scripts.audit_training_escape import fork_env, physical_trace
from scripts.diagnose_replay_mixed_windows import predict, rollout
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore


LOOKBACK = 48
CHECKPOINT_AFTER_STOP = 32
TRIAL_OFFSETS = (4, 8, 12, 16, 24, 32, 48)


def route_counts(env, requested):
    obs = rebase(env.episode.samples[env.frame].observation, env.position)
    allowed = set(feasible_actions(obs, HOLD))
    layers = []
    best = {}
    for frames, paths in feasible_prefixes(env.scene, env.belief, (0, 1), PADDING):
        counts = {}
        for actions, clearance in paths:
            action = actions[0]
            if action not in allowed:
                continue
            counts[action] = counts.get(action, 0) + 1
            if frames == SHORT_HORIZON and (action not in best or clearance > best[action][1]):
                best[action] = (actions, clearance)
        layers.append({'frames': frames, 'sequences': sum(counts.values()), 'by_first_action': counts})
    row = {'frame': env.frame, 'position': env.position.tolist(), 'baseline_request': requested,
           'planner_directions': sorted({choice.route.actions[0] for choice in env.choices}),
           'danger': env.danger, 'layers': layers}
    return row, best


def capture(anchor, policy, reference, stop):
    env, obs = restore(anchor, duration=None)
    env.fixed_delay = reference['delay']
    history = {row['frame']: row for row in reference['history']}
    rows, snapshots = [], {}
    while env.frame <= stop:
        expected = history.get(env.frame)
        action = expected['request'] if expected else None
        if env.frame >= stop - LOOKBACK:
            row, best = route_counts(env, action)
            rows.append(row)
            if stop - env.frame in TRIAL_OFFSETS:
                snapshots[env.frame] = (fork_env(env), action, best, obs)
        if env.frame == stop:
            break
        obs, _, done, _, _ = env.step(predict(policy, obs))
        if done or expected is None or physical_trace(env.history[-1:]) != physical_trace([expected]):
            raise AssertionError(f'Reference drift before {stop}, at {env.frame}')
    return rows, snapshots


def trial(origin, obs, policy, actions, clearance, limit):
    branch = fork_env(origin)
    branch.limit = limit
    branch.min_gap = float('inf')
    branch.padded_contact = branch.last_padded_contact = None
    branch.history = []
    slot = next((i for i, choice in enumerate(branch.choices) if choice.route.actions[0] == actions[0]), None)
    offered = slot is not None
    if not offered:
        # Only the first hold is executed; the full sequence certifies eligibility.
        target = np.asarray(branch.position)
        branch.choices = (Choice(Route((), tuple(target), actions), clearance - PADDING, 0., target),)
        slot = 0
    result = rollout(branch, obs, policy, first_slot=slot)
    return {'frame': origin.frame, 'request': actions[0], 'offered_by_planner': offered,
            'certificate': actions, 'certificate_clearance_px': clearance if np.isfinite(clearance) else None,
            'through_frame': branch.frame, 'outcome': result['outcome'],
            'min_clearance_px': result['min_clearance_px'],
            'first_below_padding_frame': result['first_below_padding_frame']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--terminal', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    paths = {'old': args.old, 'paired': args.paired}
    sources = {str(p): digest(p) for p in [BASE, *TRAIN, *paths.values(), args.anchors, args.terminal]}
    anchors = json.loads(args.anchors.read_text(encoding='utf-8'))
    terminal = json.loads(args.terminal.read_text(encoding='utf-8'))
    if any(sources.get(path) != value for report in (anchors, terminal)
           for path, value in report['sources'].items() if path in sources):
        raise ValueError('Source hashes differ')
    episodes = [ep for p in TRAIN for ep in load_capture(p, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policies = {name: PPO.load(path, device='cpu') for name, path in paths.items()}
    cases = []
    for anchor in anchors['anchors']:
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Stopped before anchor')
        for case in terminal['cases']:
            if (case['source'], case['episode']) != (anchor['source'], anchor['episode']):
                continue
            name, delay, stop = case['policy'], case['delay'], case['frame']
            comparison = next(label for label in policies if label != name)
            scans = {}
            for label in (name, comparison):
                reference = next(row for row in anchor['policies'][label]['episodes'] if row['delay'] == delay)
                scans[label], snapshots = capture(env, policies[label], reference, stop)
                if label == name:
                    controls = snapshots
            expected = [row['masked_routes'] for row in case['layers']
                        if row['input'] == 'belief' and row['margin_px'] == PADDING]
            if [row['sequences'] for row in scans[name][-1]['layers']] != expected:
                raise AssertionError('Exact terminal counts changed')
            print(case['source'], delay, stop, 'scan complete', flush=True)
            branches, full = [], []
            checkpoint = min(stop + CHECKPOINT_AFTER_STOP, env.episode.end)
            for frame, (origin, baseline, best, obs) in sorted(controls.items(), reverse=True):
                for action, (actions, clearance) in sorted(best.items()):
                    if action != baseline:
                        branches.append(trial(origin, obs, policies[name], actions, clearance, checkpoint))
                print('Trials through', frame, len(branches), flush=True)
            # Confirm the latest successful fork to the end of the trace.
            safe = [row for row in branches if row['through_frame'] == checkpoint
                    and row['outcome'] in ('window_end', 'trace_end')
                    and row['first_below_padding_frame'] is None]
            if safe:
                selected = max(safe, key=lambda row: (row['frame'], row['min_clearance_px']))
                origin, _, best, obs = controls[selected['frame']]
                actions, clearance = best[selected['request']]
                full.append(trial(origin, obs, policies[name], actions, clearance, env.episode.end))
            cases.append({'source': case['source'], 'episode': case['episode'], 'delay': delay,
                          'stop': stop, 'failed_policy': name, 'comparison_policy': comparison,
                          'scan': scans, 'checkpoint': checkpoint, 'trials': branches, 'full_trace': full})
            print('Safe checkpoint trials:', len(safe), 'full trace:', full, flush=True)
    if any(digest(Path(p)) != value for p, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'lookback_frames': LOOKBACK, 'trial_offsets': TRIAL_OFFSETS,
                   'cases': cases}, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
