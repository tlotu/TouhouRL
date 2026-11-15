"""Compare planners at recorded laser decisions and export a standalone replay."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agent.forward_model import scene_from_observation
from agent.replay_data import decision_record
from agent.short_planner import HOLD, LONG_HORIZON, SHORT_HORIZON, SearchBudget, plan
from scripts.probe_short_planning import _timings, _verify
from scripts.replay_planning import write_replay


MODES = {
    'short_ttc': {},
    'adaptive_ttc': {'adaptive': True},
    'budgeted_adaptive_ttc': {'adaptive': True, 'budget': SearchBudget(900, 8.)},
}


def read_windows(path, windows):
    """Keep the selected holds and their immediate predecessor for metadata alignment."""
    digest = hashlib.sha256()
    selected = [[] for _ in windows]
    ends = {}
    previous = {}
    with path.open('rb') as stream:
        raw = next(stream)
        digest.update(raw)
        header = json.loads(raw)
        if header.get('schema') != 'touhourl-static-laser-state-v1' or header.get('hold_frames') != HOLD:
            raise ValueError('Unsupported trace timing/schema')
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row['event'] == 'episode_end':
                ends[row['episode']] = {key: row[key] for key in ('reason', 'elapsed_frames', 'deaths')}
            if row['event'] != 'transition':
                continue
            episode, info = row['episode'], row['info']
            start = info['elapsed_frames'] - info['delta_frames']
            prior = previous.get(episode)
            for index, (ep, first, last) in enumerate(windows):
                if ep == episode and first <= start < last + HOLD:
                    selected[index].append((start, row, prior))
            previous[episode] = row
    return selected, ends, {'path': str(path.resolve()), 'sha256': digest.hexdigest(),
                            'task': header.get('task'), 'model_sha256': header.get('model_sha256'),
                            'candidate_slack': header.get('candidate_slack', 1.)}


def recorded_decisions(rows, last):
    for index, (start, row, prior) in enumerate(rows):
        if start > last or row['commitment_active']:
            continue
        scene, metadata = scene_from_observation(row['observation'])
        aligned = (prior is not None and prior['step'] + 1 == row['step']
                   and prior['info']['elapsed_frames'] == start
                   and prior['next_observation'] == row['observation'])
        info = prior['info'] if aligned else {}
        path = [scene.player]
        death = None
        elapsed = start
        before = prior if aligned else None
        for hold_start, step, _ in rows[index:]:
            if hold_start >= start + HOLD or hold_start != elapsed:
                break
            if step is not row and (not step['commitment_active'] or step['action'] != row['action']
                                    or before['next_observation'] != step['observation']):
                break
            elapsed = step['info']['elapsed_frames']
            if before is not None and step['info']['deaths'] > before['info']['deaths']:
                death = elapsed
                break
            px, py, *_ = step['next_observation']['player']
            path.append(((px + 1) * 192, (py + 1) * 224))
            before = step
        warnings = [laser.on for laser in scene.lasers if laser.off > laser.on and laser.on > 0]
        yield scene, prior['action'] if aligned else 0, {
            'step': row['step'], 'frame': start, 'recorded_action': row['action'],
            'recorded_path': path, 'death_observed': death,
            'dropped_circles': info.get('dropped_bullets'),
            'saturated_circles': metadata['saturated_circle_rows'],
            'radius_fallback_count': info.get('radius_fallback_count'),
            'first_light': min(warnings, default=None), 'warning_lasers': len(warnings),
            'active_lasers': sum(laser.on == 0 < laser.off for laser in scene.lasers),
        }


def compare_window(rows, window, outcome):
    episode, first, last = window
    key = f'Episode {episode + 1}, frames {first}-{last}'
    frames_by_mode = {name: [] for name in MODES}
    for scene, previous, recorded in recorded_decisions(rows, last):
        geometry = asdict(scene)
        for mode, settings in MODES.items():
            started = time.perf_counter()
            result = plan(scene, previous_action=previous, **settings)
            elapsed_ms = (time.perf_counter() - started) * 1000
            route = result.selected
            if route is not None:
                _verify(scene, route)
            frame = decision_record(scene, recorded['frame'], result=result,
                                    action=route.actions[0] if route else None)
            frame.update(recorded, scene=geometry, horizon=result.horizon,
                         safe_frames=len(route.actions) * HOLD if route else 0,
                         exit_slack=result.hint.slack_frames if result.hint else None,
                         planner_ms=elapsed_ms)
            frames_by_mode[mode].append(frame)
    if not frames_by_mode['short_ttc']:
        raise ValueError(f'No fresh decisions in {key}')
    tracks = [{'label': f'{key} / {mode}', 'trace_window': key, 'frames': frames,
               'source': 'Recorded positions and hazards. Green routes are offline predictions.',
               'outcome': (f"Recorded outcome: {outcome['reason']} at frame {outcome['elapsed_frames']}"
                           if outcome else 'Recorded outcome unavailable')}
              for mode, frames in frames_by_mode.items()]
    return tracks, {mode: summarize(frames) for mode, frames in frames_by_mode.items()}


def summarize(frames):
    def recorded_clear(frame):
        return frame['directions'][frame['recorded_action']]['status'] == 'clear'

    return {'decisions': len(frames), 'long_requested': sum(f['horizon'] == LONG_HORIZON for f in frames),
            'prefix_at_least_16': sum(f['safe_frames'] >= SHORT_HORIZON for f in frames),
            'no_proposal': sum(f['action'] is None for f in frames),
            'changed_proposal': sum(f['action'] not in (None, f['recorded_action']) for f in frames),
            'recorded_move_rejected_by_model': sum(not recorded_clear(f) for f in frames),
            'clear_alternative_when_recorded_move_rejected': sum(f['action'] is not None and not recorded_clear(f)
                                                                for f in frames),
            'stop_reasons': dict(Counter(f['stop_reason'] for f in frames)),
            'timing': _timings([f['planner_ms'] / 1000 for f in frames])}


def parse_window(value):
    try:
        episode, first, last = map(int, value.split(':'))
        if min(episode, first) >= 0 and first <= last:
            return episode, first, last
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('Expected EPISODE:FIRST:LAST with zero-based episode and inclusive frames')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--window', type=parse_window, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.window)) != len(args.window):
        parser.error('Windows must be unique')
    samples, ends, source = read_windows(args.trace, args.window)
    report = {'source': source, 'mode_settings': {name: {k: asdict(v) if isinstance(v, SearchBudget) else v
                                                      for k, v in settings.items()}
                                                  for name, settings in MODES.items()},
              'limitations': ['Offline proposals share pre-action observations from the recorded trajectory.',
                              'Future births and omitted circles are not modeled. Hitboxes and timing are approximate.',
                              'Frozen static HRL manager and static subgoals are outside this dynamic trace comparison.',
                              'Planner time excludes decoding, replay telemetry, collision verification and game IO.'],
              'tracks': [], 'summary': {}}
    for window, rows in zip(args.window, samples):
        tracks, summary = compare_window(rows, window, ends.get(window[0]))
        report['tracks'].extend(tracks)
        report['summary'][tracks[0]['trace_window']] = summary
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / 'report.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    replay = args.output / 'replay.html'
    write_replay(report, replay)
    lines = [f'Replay: {replay.resolve()}', f'Report: {target.resolve()}']
    for window, modes in report['summary'].items():
        for name, stats in modes.items():
            lines.append(f"{window} / {name}: {stats['prefix_at_least_16']}/{stats['decisions']} "
                         f"prefixes >= 16f, p95 {stats['timing']['p95_ms']:.2f} ms")
    (args.output / 'replay.console.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
