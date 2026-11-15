"""Roll alternative player paths through sampled laser and bullet geometry."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from agent.forward_model import ForwardModel, Scene, scene_from_observation
from agent.replay_data import decision_record
from agent.short_planner import HOLD, LONG_HORIZON, VECTORS, SearchBudget, plan
from agent.swept_collision import SweptCollision
from scripts.probe_planning import laser_regions
from scripts.replay_laser_trace import parse_window, read_windows
from scripts.replay_planning import write_replay


MARGINS = (0, 1, 2)
MODES = {
    'short_ttc': {},
    'adaptive_ttc': {'adaptive': True},
    'edge_budget_ttc': {'adaptive': True, 'budget': SearchBudget(900)},
    'checked_route_ttc': {'adaptive': True},
}


@dataclass(frozen=True)
class Sample:
    frame: int
    end: int
    scene: Scene
    recorded_action: int
    recorded_endpoint: tuple | None
    fresh: bool
    dropped: int | None


def samples_from_rows(rows, start, stop):
    samples = []
    expected = start
    for frame, row, prior in rows:
        if frame >= stop:
            break
        info = row['info']
        end = info['elapsed_frames']
        if frame != expected or not frame < end <= stop:
            raise ValueError('Window must have contiguous samples and end at a logged boundary')
        if prior is None or prior['info']['elapsed_frames'] != frame or prior['next_observation'] != row['observation']:
            raise ValueError('Window requires an aligned pre-action observation')
        scene, _ = scene_from_observation(row['observation'])
        endpoint = None
        if info['deaths'] == prior['info']['deaths']:
            x, y, *_ = row['next_observation']['player']
            endpoint = ((x + 1) * 192, (y + 1) * 224)
        samples.append(Sample(frame, end, scene, row['action'], endpoint, not row['commitment_active'],
                              prior['info'].get('dropped_bullets')))
        expected = end
    if expected != stop or not samples or not samples[0].fresh:
        raise ValueError('Window must start at a fresh decision and reach its declared end')
    return samples


def path_for_actions(scene, actions):
    moves = VECTORS[list(actions)] * (scene.speed * HOLD)
    return np.vstack((scene.player, scene.player + moves.cumsum(axis=0)))


def checked_suffix(scene, actions):
    """Recheck the remaining motion with current hazards before retaining it."""
    if not actions:
        return None
    points = path_for_actions(scene, actions)
    model = SweptCollision(scene)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if not model.edges([a], [b], index * HOLD, HOLD)[0][0]:
            return None
    return points.tolist()


def forecast_gap(scene, endpoint, duration):
    safe, gaps = ForwardModel(scene).edges([scene.player], [endpoint], 0, duration)
    gap = float(gaps[0])
    return bool(safe[0]), gap if math.isfinite(gap) else None


def roll(samples, mode, *, keep_first=False):
    origin = samples[0]
    position = origin.scene.player
    positions = {origin.frame: position}
    frames, contacts = [], {margin: None for margin in MARGINS}
    minimum_gap = None
    action = previous = 0
    checked = retained = 0
    decision_times = []
    remaining = ()
    reason = 'window_end'
    elapsed = origin.frame
    for sample in samples:
        scene = replace(sample.scene, player=position)
        if (sample.frame - origin.frame) % HOLD == 0:
            result, suffix = None, None
            started = time.perf_counter()
            if mode == 'recorded' or (keep_first and sample is origin):
                action = sample.recorded_action
                decision = 'Recorded action'
            else:
                if mode == 'checked_route_ttc' and remaining:
                    suffix = checked_suffix(scene, remaining)
                    checked += 1
                if suffix is not None:
                    action, *remaining = remaining
                    retained += 1
                    decision = 'Rechecked prior route'
                else:
                    result = plan(scene, previous_action=previous, **MODES[mode])
                    route = result.selected
                    action = route.actions[0] if route else None
                    remaining = route.actions[1:] if route and len(route.actions) * HOLD == LONG_HORIZON else ()
                    decision = 'New search'
            planner_ms = (time.perf_counter() - started) * 1000 if decision != 'Recorded action' else None
            if planner_ms is not None:
                decision_times.append(planner_ms)
            frame = decision_record(scene, sample.frame, result=result, action=action)
            frame.update(scene=asdict(scene), recorded_action=sample.recorded_action, decision_source=decision,
                         dropped_circles=sample.dropped, branch_path=[position], planner_ms=planner_ms)
            if suffix is not None:
                frame['route_points'] = suffix
            frames.append(frame)
            if action is None:
                reason = 'no_plan'
                break
            previous = action
        if (sample.frame - origin.frame) % HOLD + sample.end - sample.frame > HOLD:
            raise ValueError('A sample crosses a simulated action boundary')
        if mode == 'recorded':
            if sample.recorded_endpoint is None:
                frame['death_observed'] = sample.end
                reason = 'recorded_death_endpoint_missing'
                break
            endpoint = sample.recorded_endpoint
        else:
            endpoint = tuple(position + VECTORS[action] * scene.speed * (sample.end - sample.frame))
        safe, gap = forecast_gap(scene, endpoint, sample.end - sample.frame)
        if gap is not None:
            minimum_gap = gap if minimum_gap is None else min(minimum_gap, gap)
            for margin in MARGINS:
                if gap <= margin and contacts[margin] is None:
                    contacts[margin] = [sample.frame, sample.end]
        elapsed = sample.end
        position = endpoint
        frame['branch_path'].append(position)
        if mode != 'recorded' and not safe:
            reason = 'model_contact' if gap is not None and gap <= 0 else 'out_of_bounds'
            frame['branch_event'] = f'{reason} in frames {sample.frame}–{sample.end}'
            break
        positions[elapsed] = position
    timing = {'samples': len(decision_times), 'p95_ms': float(np.percentile(decision_times, 95)),
              'max_ms': max(decision_times)} if decision_times else None
    return frames, positions, {'stop': reason, 'evaluated_to_frame': elapsed, 'planner_timing': timing,
                               'first_contact_interval_by_margin_px': contacts, 'minimum_gap_px': minimum_gap,
                               'route_rechecks': checked, 'route_retained': retained}


def compare_window(samples, window, keep_first):
    episode, start, stop = window
    label = f'Episode {episode + 1}, frames {start}-{stop}'
    scene = samples[0].scene
    events = {laser.on for laser in scene.lasers if start + laser.on <= stop}
    peak = max(events, key=lambda t: (sum(l.on <= t < l.off for l in scene.lasers), -t), default=None)
    area, peak_frame = None, None
    if peak is not None:
        peak_frame = int(start + peak)
        area, _ = laser_regions(scene, peak)
    tracks, stats = [], {}
    branches = [(mode, mode, False) for mode in ('recorded', *MODES)]
    if keep_first:
        branches.append(('recorded first, then short_ttc', 'short_ttc', True))
    for name, mode, forced in branches:
        frames, positions, summary = roll(samples, mode, keep_first=forced)
        room = area(positions[peak_frame]) if area and peak_frame in positions else None
        summary.update(peak_laser_frame=peak_frame, peak_room_area_px2=room)
        stats[name] = summary
        outcome = f"{summary['stop']} · evaluated through frame {summary['evaluated_to_frame']}"
        if room is not None:
            outcome += f' · laser-only area at frame {peak_frame}: {room:.0f} px²'
        tracks.append({'label': f'{label} / {name}', 'counterfactual_window': label, 'frames': frames,
                       'source': 'Recorded trajectory.' if mode == 'recorded'
                       else 'Simulated player with refreshed recorded hazards. No game outcome.', 'outcome': outcome})
    return label, tracks, stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--window', type=parse_window, action='append', required=True,
                        help='Zero-based episode, start frame and final sample endpoint')
    parser.add_argument('--keep-first-control', action='store_true', help='Also keep the first action before short TTC')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if len(set(args.window)) != len(args.window):
        parser.error('Windows must be unique')
    windows, _, source = read_windows(args.trace, args.window)
    report = {'source': source, 'tracks': [], 'summary': {},
              'configuration': {'hold_frames': HOLD, 'padding_px': MARGINS,
                                'adaptive_edge_budget': MODES['edge_budget_ttc']['budget'].max_edges},
              'limitations': [
        'The player branches. Later hazard samples are taken from the original trajectory.',
        'Hazards may respond to the original player, and bullet omissions can depend on position.',
        'Within each sample interval circles are linear and laser timing follows the decoded phase.',
        'Branch movement uses nominal focused speed and fixed four-frame holds.',
        'Collision padding is a deterministic sensitivity check, not a probability estimate.',
        'The final recorded step after a death has no verified player endpoint.',
        'Planner timings exclude acquisition, independent segment checks, telemetry and game IO.',
    ]}
    for window, rows in zip(args.window, windows):
        label, tracks, stats = compare_window(samples_from_rows(rows, *window[1:]), window, args.keep_first_control)
        report['tracks'].extend(tracks)
        report['summary'][label] = stats
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
    write_replay(report, args.output / 'replay.html')
    lines = [f'Replay: {(args.output / "replay.html").resolve()}']
    for label, branches in report['summary'].items():
        lines.extend(f'{label} / {mode}: {summary}' for mode, summary in branches.items())
    (args.output / 'replay.console.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
