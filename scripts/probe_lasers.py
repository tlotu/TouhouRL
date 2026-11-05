"""Read candidate laser slots alongside circular bullets; never send input."""
import argparse
from collections import Counter
import sys
import time

from env.laser_probe import capture, require_layout, LAYOUT
from env.memory_reader import SnapshotRaceError
from env.windows import connect
from scripts.diagnostics import DiagnosticLog, timing_clock_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--spell', required=True)
    parser.add_argument('--difficulty', required=True, choices=('Easy','Normal','Hard','Lunatic'))
    parser.add_argument('--seconds', type=float, default=45.)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--delay', type=float, default=5.)
    parser.add_argument('--raw-pool', action='store_true')
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    if not (args.spell.strip() and 1 <= args.seconds <= 300 and .02 <= args.interval <= 1
            and 0 <= args.delay <= 30):
        parser.error('Require a spell label, seconds 1-300, interval .02-1, delay 0-30')
    with DiagnosticLog(args.output) as log:
        reader,pid,build = connect(args.pid)
        samples=races=peak_lasers=peak_bullets=invalid=combat_samples=0
        phases=Counter()
        try:
            require_layout(reader,build)
            log.emit({'event':'connected','pid':pid,'build_md5':build,
                'collection':'laser_layout_probe','read_only':True,'verified':False,
                'layout':LAYOUT,'spell':args.spell,'difficulty':args.difficulty,
                'identity_source':'manual','raw_pool':args.raw_pool,
                'timing_clock':timing_clock_info(),'seconds':args.seconds,'interval':args.interval})
            print(f'Switch to the selected spell within {args.delay:g} seconds. Play/retry manually. '
                  'No keys, shooting, retries or pauses will be sent. Phase labels are unverified.',file=sys.stderr)
            time.sleep(args.delay)
            start = time.perf_counter()
            while time.perf_counter()-start < args.seconds:
                begun=time.perf_counter()
                try:
                    sample,record=capture(reader,raw_pool=args.raw_pool)
                except SnapshotRaceError as exc:
                    races+=1
                    log.emit({'event':'acquisition_skipped','type':type(exc).__name__,
                              'message':str(exc),'elapsed_seconds':time.perf_counter()-start},console=False)
                else:
                    samples+=1
                    combat_samples+=int(record['pool_sampled'])
                    peak_lasers=max(peak_lasers,record['candidate_count'])
                    peak_bullets=max(peak_bullets,len(sample.bullets))
                    phases.update(str(r['phase_raw'])for r in record['lasers'])
                    invalid+=sum(bool(r['issues'])for r in record['lasers'])
                    log.emit({'event':'laser_probe_sample','sample':samples-1,
                        'elapsed_seconds':time.perf_counter()-start,
                        'capture_seconds':time.perf_counter()-begun,
                        'scene':sample.scene,'wanted_scene':sample.wanted_scene,
                        'running':sample.running,'dying':sample.dying,'deaths':sample.deaths,
                        'result':sample.result_menu.outcome if sample.result_menu else None,
                        'player':[sample.x,sample.y] if sample.running else None,
                        'enemies':[e.record()for e in sample.enemies],
                        'bullet_count':len(sample.bullets),
                        'bullets':[{'slot':b.slot,'x':b.x,'y':b.y,'vx':b.vx,'vy':b.vy,
                                    'radius':b.radius,'state':b.state,'radius_estimated':b.radius_estimated}
                                   for b in sample.bullets], **record},console=False)
                time.sleep(max(0,args.interval-(time.perf_counter()-begun)))
            log.emit({'event':'laser_probe_summary','samples':samples,
                'combat_samples':combat_samples,
                'skipped_races':races,'peak_laser_candidates':peak_lasers,
                'phase_raw_histogram':dict(phases),'peak_bullets':peak_bullets,
                'invalid_candidate_records':invalid,'verified':False,
                'status':('candidates_captured' if peak_lasers else 'no_candidates_observed'
                          if combat_samples else 'no_combat_samples' if samples else 'no_valid_samples')})
            print(f'Read-only capture complete: {samples} accepted, {races} skipped, '
                  f'peak laser candidates={peak_lasers}, peak bullets={peak_bullets}. '
                  'Layout and phase hints remain unverified.')
        finally:
            reader.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped read-only capture.',file=sys.stderr)
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        sys.exit(1)
