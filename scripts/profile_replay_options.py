"""Measure candidate generation and verification inside a projected forecast."""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import json

import numpy as np

from agent import replay_hrl
from agent.direction_buffer import DirectionBuffer
from agent.short_planner import plan
from scripts.audit_replay_replanning_scores import forecast
from scripts.train_replay_hrl import BASE, TRAIN, digest, load_capture


def independent_beams(model, state, actions, delay, width):
    groups = {}
    checked = 0
    for action in actions:
        result = plan(model.scene, previous_action=state.requested, width=width, smooth=False,
                                 input_buffer=state, switch_delay=delay, ttc_weight=0.,
                                 collision_model=model, first_action=action)
        complete = [r for r in result.candidates if len(r.actions) * replay_hrl.HOLD == replay_hrl.SHORT_HORIZON]
        if complete:
            groups[action] = complete
        checked += result.checked_edges
    return groups, checked


class PhaseTimes:
    """Collect nested wall-clock timings for one decision at a time."""

    def __init__(self):
        self.elapsed = Counter()
        self.calls = Counter()

    def timed(self, fn, label):
        def run(*args, **kwargs):
            start = perf_counter()
            result = fn(*args, **kwargs)
            self.elapsed[label] += (perf_counter() - start) * 1000
            self.calls[label] += 1
            return result
        return run

    @contextmanager
    def options(self, worker):
        with (patch.object(replay_hrl, 'plan', self.timed(replay_hrl.plan, 'plan')),
              patch.object(replay_hrl, 'forced_beams', self.timed(replay_hrl.forced_beams, 'forced_plan')),
              patch.object(replay_hrl, 'robust_routes', self.timed(replay_hrl.robust_routes, 'verify')),
              patch.object(worker, 'options', self.timed(worker.options, 'options'))):
            yield

    def take(self):
        row = {'ms': dict(self.elapsed), 'calls': dict(self.calls)}
        self.elapsed.clear()
        self.calls.clear()
        return row


def measure(origin, worker):
    timer = PhaseTimes()
    with timer.options(worker):
        result = forecast(origin, None, worker)
    return result, timer.take()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    worker = replay_hrl.ReplayWorker(lambda obs: np.zeros(9), prune_failed=True)
    rows = []
    for path, frame in zip(TRAIN, (928, 392)):
        episode = next(e for e in load_capture(path, digest(BASE)) if e.number == 2)
        origin = SimpleNamespace(scene=episode.samples[frame].scene, frame=frame, episode=episode,
                                 belief=frozenset([DirectionBuffer()]))
        reference = forecast(origin, None, worker)
        batched = replay_hrl.forced_beams
        def compare(*inputs):
            actual = batched(*inputs)
            if actual != independent_beams(*inputs):
                raise AssertionError('Forced candidates or edge counts differ')
            return actual
        with patch.object(replay_hrl, 'forced_beams', compare):
            forecast(origin, None, worker)
        for repeat in range(3):
            for grouped in ((False, True) if repeat % 2 == 0 else (True, False)):
                with patch.object(replay_hrl, 'forced_beams', batched if grouped else independent_beams):
                    result, row = measure(origin, worker)
                if result != reference:
                    raise AssertionError('Timed forecast differs')
                rows.append(dict(source=str(path), frame=frame, repeat=repeat, grouped=grouped, **row))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)


if __name__ == '__main__':
    main()
