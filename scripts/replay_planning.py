"""Export a standalone, pausable HTML replay from an offline report."""
import argparse
import json
from pathlib import Path


def replay_tracks(report):
    if 'tracks' in report:
        return report['tracks']
    tracks = []
    for case in report['micro']:
        for mode, result in case['modes'].items():
            outcome = 'Pass' if result['pass'] else 'Missed objective'
            arrival = result.get('goal_arrival_frame')
            if arrival is not None:
                outcome += f'. Goal reached at frame {arrival}'
            tracks.append({'label': f"{case['name']} / {mode}", 'scene': case['scene'],
                           'goal': case['goal'], 'frames': [d['replay'] for d in result['decisions']],
                           'outcome': outcome,
                           'source': 'Model rollout'})
    for case in report['snapshots']:
        for mode, result in case['modes'].items():
            frame = {**result['replay'], 'recorded_action': case['metadata'].get('recorded_action')}
            tracks.append({'label': f"{case['name']} / {mode}", 'scene': case['scene'],
                           'frames': [frame], 'frame_offset': case['metadata']['start_frame'],
                           'source': 'Frozen trace snapshot. Displayed route is a prediction.'})
    for track in report.get('diagnostic_tracks', []):
        tracks.append(track)
    return tracks


def write_replay(report, target):
    tracks = replay_tracks(report)
    if not tracks or any(not track['frames'] for track in tracks):
        raise ValueError('Report has no replay frames')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    payload = json.dumps(tracks, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    with Path(target).open('x', encoding='utf-8') as stream:
        stream.write(template.replace('__REPLAY_DATA__', payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = args.output or args.report.with_name('replay.html')
    write_replay(json.loads(args.report.read_text(encoding='utf-8')), target)
    print(target.resolve())


if __name__ == '__main__':
    main()
