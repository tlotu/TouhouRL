"""Paired component timings at recorded player positions on two training captures."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agent.direction_buffer import DirectionBuffer
from agent.replay_hrl import PADDING, ReplayWorker
from agent.swept_collision import SweptCollision
from scripts.audit_replay_replanning_scores import forecast
from scripts.audit_replay_terminal_routes import feasible_prefixes
from scripts.train_replay_hrl import BASE, TRAIN, digest, load_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    worker = ReplayWorker(lambda obs: np.zeros(9), prune_failed=True)
    for path, frame in zip(TRAIN, (928, 392)):
        episode = next(ep for ep in load_capture(path, digest(BASE)) if ep.number == 2)
        sample = episode.samples[frame]
        belief = frozenset([DirectionBuffer()])
        origin = SimpleNamespace(scene=sample.scene, frame=frame, episode=episode, belief=belief)
        for repeat, order in enumerate(((False, True), (True, False))):
            for fast in order:
                start = perf_counter()
                for _, layer in feasible_prefixes(sample.scene, belief, (0, 1), PADDING,
                                                  final_counts=fast, batch_size=1024 if fast else 128,
                                                  model_factory=lambda s: SweptCollision(s, clearance_cap=1e-6)):
                    pass
                counts = layer if fast else np.bincount([a[0] for a, _ in layer], minlength=9).tolist()
                count_ms = (perf_counter() - start) * 1000
                with patch.object(worker, 'options', wraps=worker.options) as calls:
                    start = perf_counter()
                    result = forecast(origin, None, worker, reuse_options=fast)
                    forecast_ms = (perf_counter() - start) * 1000
                if repeat == 0 and not fast:
                    reference = counts, result
                if (counts, result) != reference:
                    raise AssertionError('Component outputs differ')
                rows.append(dict(source=str(path), frame=frame, fast=fast, repeat=repeat, counts=counts,
                                 forecast=result, count_ms=count_ms, forecast_ms=forecast_ms, options=calls.call_count))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
