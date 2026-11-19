"""Swap policies after shared-state divergence in audited replay branches."""
import argparse
import json
from pathlib import Path

import torch

from agent.replay_hrl import HOLD, ReplayWorker
from scripts.audit_training_escape import fork_env, physical_trace
from scripts.diagnose_replay_mixed_windows import predict, rollout
from scripts.train_replay_hrl import BASE, TRAIN, PPO, PPOBackend, digest, load_capture
from scripts.train_replay_mixed_windows import CoverageEnv, restore

PREFIX_HOLDS = (1, 4, 8)


def advance(env, obs, policy, expected):
    obs, _, done, _, _ = env.step(predict(policy, obs))
    if done or physical_trace(env.history[-1:]) != physical_trace([expected]):
        raise AssertionError(f'Prefix did not reproduce at frame {expected["frame"]}')
    return obs


def handoffs(env, obs, policies, references):
    old, paired = (references[name]['history'] for name in ('old', 'paired'))
    divergence = next((i for i, (a, b) in enumerate(zip(old, paired))
                       if a['request'] != b['request']), None)
    if divergence is None:
        if physical_trace(old) != physical_trace(paired):
            raise AssertionError('Unmatched trajectories without a different request')
        return None, []
    for expected in old[:divergence]:
        obs = advance(env, obs, policies['old'], expected)
    decision = {'frame': env.frame, 'position': env.position.tolist(), 'danger': env.danger,
                'distinct_moves': len({choice.route.actions[0] for choice in env.choices}),
                'choices': {}}
    for name, policy in policies.items():
        slot = predict(policy, obs)
        choice = env.choices[slot]
        decision['choices'][name] = {'slot': slot, 'move': choice.route.actions[0],
                                     'leaf_ttc': choice.ttc, 'clearance_px': choice.gap}
    swaps = []
    for before, after in [('old', 'paired'), ('paired', 'old')]:
        branch, branch_obs = fork_env(env), obs
        expected = references[before]['history'][divergence:]
        for i, move in enumerate(expected[:max(PREFIX_HOLDS)], start=1):
            branch_obs = advance(branch, branch_obs, policies[before], move)
            if i not in PREFIX_HOLDS:
                continue
            result = rollout(fork_env(branch), branch_obs, policies[after])
            end = result['start'] + result['frames']
            swaps.append({'prefix_policy': before, 'suffix_policy': after, 'holds': i,
                          'handoff_frame': branch.frame, 'through_frame': end, **result})
            print(env.episode.source, env.fixed_delay, env.frame, before, after, i,
                  end, result['outcome'], flush=True)
    return decision, swaps


def isolate_later_move(env, obs, policies, reference, case):
    """Isolate the first changed request in a handoff that altered the outcome."""
    mixed = next((row for row in case['handoffs'] if row['prefix_policy'] == 'old'
                  and row['outcome'] != reference['outcome']), None)
    if mixed is None:
        return None
    for expected, alternate in zip(reference['history'], mixed['history']):
        if expected['request'] != alternate['request']:
            break
        obs = advance(env, obs, policies['old'], expected)
    else:
        raise AssertionError('Changed outcome without a different request')
    frame = env.frame
    slot = predict(policies['paired'], obs)
    if frame != alternate['frame'] or env.choices[slot].route.actions[0] != alternate['request']:
        raise AssertionError('Changed request did not reproduce')
    result = rollout(env, obs, policies['old'], first_slot=slot)
    end = result['start'] + result['frames']
    print(case['anchor'], case['delay'], frame, expected['request'], alternate['request'], end,
          result['outcome'], flush=True)
    return {'frame': frame, 'old_move': expected['request'], 'replacement_move': alternate['request'],
            'through_frame': end, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchors', type=Path, required=True)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--isolate-handoffs', type=Path, help='Isolate later moves from a prior handoff report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    paths = {'old': args.old, 'paired': args.paired}
    sources = {str(path): digest(path) for path in [BASE, *TRAIN, args.anchors, *paths.values()]}
    report = json.loads(args.anchors.read_text(encoding='utf-8'))
    if any(report['sources'].get(str(path)) != sources[str(path)]
           for path in [BASE, *TRAIN, *paths.values()]):
        raise ValueError('Anchor sources do not match')
    prior = None
    if args.isolate_handoffs:
        prior = json.loads(args.isolate_handoffs.read_text(encoding='utf-8'))
        if prior['sources'] != sources:
            raise ValueError('Handoff sources do not match')
        sources[str(args.isolate_handoffs)] = digest(args.isolate_handoffs)
    episodes = [ep for path in TRAIN for ep in load_capture(path, sources[str(BASE)])]
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    backend.model.policy.requires_grad_(False).eval()
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    policies = {name: PPO.load(path, device='cpu') for name, path in paths.items()}
    cases = []
    for anchor in report['anchors']:
        index = next(i for i, ep in enumerate(episodes)
                     if (ep.source, ep.number) == (anchor['source'], anchor['episode']))
        env = CoverageEnv(episodes, worker, verify=True)
        obs, _ = env.reset(seed=0, options={'episode': index, 'delay': 0})
        while env.frame < anchor['frame']:
            obs, _, done, _, _ = env.step(predict(policies['old'], obs))
            if done:
                raise AssertionError('Baseline stopped before anchor')
        for delay in (0, 1):
            references = {name: next(row for row in anchor['policies'][name]['episodes'] if row['delay'] == delay)
                          for name in policies}
            branch, branch_obs = restore(env, duration=None)
            branch.fixed_delay = delay
            if prior is not None:
                case = next(row for row in prior['cases'] if row['source'] == anchor['source']
                            and row['anchor'] == env.frame and row['delay'] == delay)
                isolated = isolate_later_move(branch, branch_obs, policies, references['old'], case)
                cases.append({'source': anchor['source'], 'anchor': env.frame, 'delay': delay,
                              'isolated_move': isolated})
                continue
            decision, swaps = handoffs(branch, branch_obs, policies, references)
            cases.append({'source': anchor['source'], 'episode': anchor['episode'], 'anchor': env.frame,
                          'delay': delay, 'divergence': decision, 'handoffs': swaps,
                          'controls': {name: {key: row[key] for key in ('frames', 'outcome', 'start',
                                                                        'first_below_padding_frame')}
                                       for name, row in references.items()}})
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise AssertionError('Source changed')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'sources': sources, 'hold_frames': HOLD, 'prefix_holds': PREFIX_HOLDS, 'cases': cases},
                  stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
