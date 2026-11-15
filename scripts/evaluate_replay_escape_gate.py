"""Compare frozen waypoint policies during forecast connectivity contractions."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from agent.replay_connectivity import forecast_horizon
from agent.replay_hrl import HOLD, ReplayManagerEnv, ReplayWorker, rebase
from scripts.train_replay_hrl import BASE, DIAGNOSTICS, TEST, PPO, PPOBackend
from scripts.train_replay_hrl import attack_position_summary, digest, load_capture


SCAN_INTERVAL = 16
MAX_LEAD = 32


@dataclass
class ContractionGate:
    next_scan: int
    until: int | None = None

    def update(self, frame, forecast):
        if self.until is not None and frame < self.until:
            return True
        self.until = None
        if frame < self.next_scan:
            return False
        self.next_scan = frame + SCAN_INTERVAL
        event = forecast()
        if (event is None or event['reason'] == 'capped'
                or not 0 <= event['start'] <= MAX_LEAD or event['end'] <= 0):
            return False
        self.until = frame + event['end']
        self.next_scan = self.until + HOLD
        return True


class GatedReplayEnv(ReplayManagerEnv):
    """Keep the original waypoint interface except during a forecast contraction."""

    def __init__(self, episodes, worker):
        super().__init__(episodes, worker, verify=True)

    def reset(self, *, seed=None, options=None):
        self.gate = ContractionGate((options or {}).get('frame', 0))
        self.scans = []
        return super().reset(seed=seed, options=options)

    def _forecast(self):
        started = time.perf_counter()
        horizon, event, areas = forecast_horizon(self.scene)
        self.scans.append({'frame': self.frame, 'horizon': horizon, 'event': event,
                           'area_now_px2': areas[0][1], 'area_min_px2': min(area for _, area in areas),
                           'ms': 1000 * (time.perf_counter() - started)})
        return event

    def _prepare(self):
        obs = super()._prepare()
        self.escape_active = bool(self.choices) and self.gate.update(self.frame, self._forecast)
        if not self.escape_active:
            return obs
        sample = self.episode.samples[self.frame]
        rebased = rebase(sample.observation, self.position)
        self.choices, edges, elapsed = self.worker.options(self.scene, rebased, self.belief,
                                                          waypoint_coverage=True)
        self.times[-1] += elapsed
        self.edges[-1] += edges
        return self._observation(rebased, sample)


def evaluate(episodes, worker, old, inside, cases):
    env = GatedReplayEnv(episodes, worker)
    rows = []
    for case in cases:
        obs, _ = env.reset(seed=0, options=case)
        switched = []
        active_frames = []
        while True:
            active = env.escape_active
            policy = inside if active else old
            slot = policy.predict(obs, deterministic=True)[0].item()
            if active:
                active_frames.append(env.frame)
                if env.choices and env.choices[slot].route.actions[0] != env.choices[0].route.actions[0]:
                    switched.append(env.frame)
            obs, _, done, _, info = env.step(slot)
            if done:
                break
        rows.append({'source': env.episode.source, 'episode': env.episode.number, 'delay': case['delay'],
                     'gate_frames': active_frames, 'nonrule_frames_in_gate': switched,
                     'scans': env.scans, **info})
    return {'completed': sum(row['outcome'] == 'trace_end' for row in rows), 'total': len(rows),
            'mean_frames': float(np.mean([row['frames'] for row in rows])),
            'gate_decisions': sum(len(row['gate_frames']) for row in rows),
            'nonrule_gate_decisions': sum(len(row['nonrule_frames_in_gate']) for row in rows),
            'forecast_p95_ms': float(np.percentile([s['ms'] for row in rows for s in row['scans']], 95)),
            'attack_position': attack_position_summary(episodes, rows), 'episodes': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--escape', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output already exists')
    torch.set_num_threads(1)
    sources = {str(path): digest(path) for path in (BASE, TEST, args.old, args.escape)}
    episodes = load_capture(TEST, sources[str(BASE)])
    backend = PPOBackend.__new__(PPOBackend)
    backend.model = PPO.load(BASE, device='cpu')
    worker = ReplayWorker(backend.action_logits, prune_failed=True)
    old = PPO.load(args.old, device='cpu')
    escape = PPO.load(args.escape, device='cpu')
    cases = {'full': [{'episode': i, 'delay': d} for i in range(len(episodes)) for d in (0, 1)],
             'early': [{'episode': 2, 'frame': f, 'delay': d} for f in DIAGNOSTICS[:3] for d in (0, 1)]}
    report = {'sources': sources, 'scan_interval_frames': SCAN_INTERVAL, 'max_lead_frames': MAX_LEAD,
              'evaluations': {}}
    for label, policy in [('old_in_gate', old), ('escape_in_gate', escape)]:
        report['evaluations'][label] = {}
        for group, trials in cases.items():
            result = evaluate(episodes, worker, old, policy, trials)
            report['evaluations'][label][group] = result
            print(label, group, f"{result['completed']}/{result['total']}",
                  f"{result['mean_frames']:.1f} frames", flush=True)
    if any(digest(Path(path)) != original for path, original in sources.items()):
        raise AssertionError('Source changed during evaluation')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
