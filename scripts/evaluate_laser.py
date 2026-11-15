"""Evaluate a frozen static-laser PPO checkpoint without training."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time

from agent.checkpoints import read_checkpoint
from agent.laser_contract import LASER_PPO_CONTRACT,LASER_SCHEMA,LASER_TASK
from agent.laser_policy import LaserRiskPolicy
from scripts.diagnostics import DiagnosticLog,run_episodes,timing_clock_info
from scripts.train_laser import LASER_CONFIG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--output',required=True)
    parser.add_argument('--pid',type=int)
    parser.add_argument('--focus-timeout',type=float,default=30.)
    parser.add_argument('--candidate-slack',type=float,default=1.,
                        help='Evaluation-only clearance loss from best, in pixels (0-8)')
    planners=parser.add_mutually_exclusive_group()
    planners.add_argument('--planner-shadow',action='store_true',
                          help='Log budgeted planner proposals alongside baseline actions')
    planners.add_argument('--planner-buffered-shadow',action='store_true',
                          help='Measure buffered planning in game while keeping baseline input')
    planners.add_argument('--planner-async-shadow',action='store_true',
                          help='Measure speculative four-frame planning while keeping baseline input')
    planners.add_argument('--planner-cache-shadow',action='store_true',
                          help='Audit the cached planner without changing actions')
    planners.add_argument('--planner-intervene',type=Path,metavar='SHADOW_JSONL',
                          help='Use guarded interventions after validating a matching ten-episode shadow log')
    parser.add_argument('--planner-switch-guard',action='store_true',
                        help='Experimental transition recheck in cached planner modes (off by default)')
    parser.add_argument('--profile-input', action='store_true',
                        help='Split foreground checks and SendInput timings in buffered shadow mode')
    parser.add_argument('--batch-multi-key-input', action='store_true',
                        help='Submit direction switches with two or more key events in one SendInput call')
    args=parser.parse_args()
    if (args.task!=LASER_TASK or args.episodes<1 or not 0<args.focus_timeout<=60
            or not math.isfinite(args.candidate_slack) or not 0<=args.candidate_slack<=8):
        parser.error('Select the declared laser task and valid episode/focus settings')
    if args.planner_switch_guard and not (args.planner_cache_shadow or args.planner_intervene):
        parser.error('Planner switch guard requires a cached planner mode')
    if args.profile_input and not (args.planner_buffered_shadow or args.planner_async_shadow):
        parser.error('Input profiling requires buffered or async planner shadow mode')
    if args.batch_multi_key_input and not args.profile_input:
        parser.error('Multi-key batching requires --profile-input in buffered planner shadow mode')
    saved=read_checkpoint(args.resume,evaluation=True)
    config=saved['config']
    if (config.get('ppo_contract')!=LASER_PPO_CONTRACT or saved['schema']!=LASER_SCHEMA
            or config.get('laser_filter')!=LASER_CONFIG):
        parser.error('Evaluation requires a matching static-laser PPO checkpoint')
    from agent.risk_contract import RISK_PPO_CONTRACT
    from scripts.train_masked import risk_config
    prior={**saved,'config':{**config,'ppo_contract':RISK_PPO_CONTRACT}}
    prior['config'].pop('laser_filter')
    try:risk_config(prior)
    except ValueError as exc:parser.error(str(exc))
    from agent.laser_backend import LaserPPOBackend
    from agent.laser_evaluation import LaserEvaluationCommitment
    from env.action_emulator import ActionEmulator
    from env.frame_stepper import ACQUISITION_CONTRACT
    from env.laser_session import LaserReader,LaserSession,LaserTouhouEnv
    from env.windows import INPUT_TIMING_CONTRACT,MULTI_KEY_INPUT_TIMING_CONTRACT,WindowsKeyboard,connect

    controller, shadow_config = LaserEvaluationCommitment, {}
    if args.planner_async_shadow:
        from agent.laser_async_shadow import ASYNC_SHADOW_CONTRACT, LaserAsyncShadowCommitment
        from agent.short_planner import HOLD
        if config['hold_frames'] != HOLD:
            parser.error('Async planner shadow requires a four-frame checkpoint')
        controller = LaserAsyncShadowCommitment
        shadow_config = {'planner_async_shadow_contract': ASYNC_SHADOW_CONTRACT}
    if args.planner_shadow or args.planner_buffered_shadow:
        from agent.laser_shadow import (BUFFERED_LEAF_CONFIG,SHADOW_CONFIG,
                                        LaserBufferedReuseCommitment,LaserShadowCommitment)
        from agent.short_planner import HOLD
        if config['hold_frames']!=HOLD:
            parser.error('Planner shadow requires a four-frame checkpoint')
        if args.planner_buffered_shadow:
            controller, shadow_config = LaserBufferedReuseCommitment, {'planner_shadow':BUFFERED_LEAF_CONFIG}
        else:
            controller, shadow_config = LaserShadowCommitment, {'planner_shadow':SHADOW_CONFIG}

    archive=Path(args.resume)/'model.zip'
    model_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    cached_mode = 'intervene' if args.planner_intervene else 'shadow' if args.planner_cache_shadow else None
    if cached_mode:
        from agent.laser_cached import CACHE_CONFIG,CachedLaserCommitment
        from agent.short_planner import HOLD
        if config['hold_frames']!=HOLD:
            parser.error('Cached planner requires a four-frame checkpoint')
        controller = CachedLaserCommitment
        shadow_config = {'planner_cache_config':CACHE_CONFIG,'planner_cache_mode':cached_mode}
        if args.planner_switch_guard:
            from agent.input_switch import SWITCH_GUARD
            shadow_config['planner_switch_guard'] = SWITCH_GUARD
        if args.planner_intervene:
            from scripts.analyze_laser_shadow import validate_cached_shadow
            shadow_config['validated_shadow_sha256'] = validate_cached_shadow(
                args.planner_intervene,model_hash,saved['build_md5'],args.candidate_slack,
                switch_guard=args.planner_switch_guard)
    with DiagnosticLog(args.output) as log:
        reader,pid,build=connect(args.pid)
        try:
            if build!=saved['build_md5'] or 'SPELL_RESULT_ACTIVE' not in reader.addresses:
                raise RuntimeError('Laser evaluation requires the matching verified game build')
            reader=LaserReader(reader,build)
            switch_direction = None
            if args.planner_switch_guard:
                from env.movement_probe import MovementProbe
                switch_direction = MovementProbe(reader,build).direction
            controls=ActionEmulator(WindowsKeyboard(pid),profile_input=args.profile_input,
                                    batch_multi_key=args.batch_multi_key_input)
            session=LaserSession(reader,controls,repeat=2,capacity=config['capacity'],
                                 max_frames=config['max_frames'],spell_retry=True)
        except BaseException:
            reader.close();raise
        with session:
            learner=LaserPPOBackend(LaserTouhouEnv(session),checkpoint=str(archive))
            if type(learner.model.policy) is not LaserRiskPolicy or learner.timesteps!=saved['timesteps']:
                raise RuntimeError('Checkpoint policy or counters differ')
            learner.enable_laser(LaserTouhouEnv(session).observation_space,config['hold_frames'])
            cached_options = {}
            if cached_mode:
                cached_options = {'sample_status':lambda:(session.sample.frame,session.sample_received_at),
                                  'read_frame':reader.frame,'intervene':cached_mode=='intervene',
                                  'read_direction':switch_direction}
            logits=learner.action_logits
            if args.planner_buffered_shadow or args.planner_async_shadow:
                def capture_started():
                    read = session.timing.get('sample_read_seconds')
                    return session.sample_received_at - read if read is not None else None

                def timed_logits(observation):
                    nonlocal logits_seconds
                    started = time.perf_counter()
                    values = learner.action_logits(observation)
                    logits_seconds = time.perf_counter() - started
                    return values

                logits_seconds = 0.
                logits = timed_logits
                cached_options['capture_started'] = capture_started
            choose=controller(logits,lambda:session.elapsed_frames,config['hold_frames'],
                              candidate_slack=args.candidate_slack,**cached_options)
            external_gate=args.candidate_slack!=config['risk_filter']['max_clearance_loss']
            policy='laser_ppo_external_slack_held' if external_gate else 'laser_ppo_deterministic_held'
            collection='laser_ppo_external_slack_evaluation' if external_gate else 'laser_ppo_evaluation'
            if args.planner_buffered_shadow:
                collection='laser_ppo_buffered_leaf_shadow'
            if args.planner_async_shadow:
                collection='laser_ppo_async_shadow'
            if cached_mode:
                collection=f'laser_ppo_cached_{cached_mode}_evaluation'
            if cached_mode == 'intervene':
                policy='laser_ppo_cached_intervention'
            if args.profile_input:
                shadow_config['input_timing_contract'] = (
                    MULTI_KEY_INPUT_TIMING_CONTRACT if args.batch_multi_key_input else INPUT_TIMING_CONTRACT)
            log.emit({'event':'connected','pid':pid,'build_md5':build,'schema':LASER_SCHEMA,
                'policy':policy,'collection':collection,'task':args.task,
                'task_verification':'manual','training':False,'checkpoint':str(Path(args.resume).resolve()),
                'model_sha256':model_hash,'timesteps':saved['timesteps'],'updates':saved['updates'],
                'hold_frames':config['hold_frames'],'repeat':2,'capacity':config['capacity'],
                'candidate_slack':args.candidate_slack,'external_candidate_gate':external_gate,
                'laser_filter':config['laser_filter'],'risk_filter':config['risk_filter'],
                'acquisition_contract':ACQUISITION_CONTRACT,'timing_clock':timing_clock_info(),**shadow_config})
            print('Select the declared spell result menu and focus the game. Frozen deterministic evaluation.',file=sys.stderr)
            if args.planner_shadow:
                print('Planner shadow: baseline actions with synchronous proposal logging.',file=sys.stderr)
            if args.planner_buffered_shadow:
                print('Buffered shadow: baseline input with rechecked routes and deadline-aware search.',file=sys.stderr)
            if cached_mode:
                print(f'Cached planner mode: {cached_mode}.',file=sys.stderr)
            if external_gate:
                print(f'External evaluation gate: candidate slack {args.candidate_slack:g} px; '
                      'not the checkpoint sampling distribution and not PPO training.',file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            previous_trace_timing = None
            previous_serialized_observation = None
            def emit(row):
                nonlocal previous_trace_timing, previous_serialized_observation
                if row['event']=='reset':
                    previous_trace_timing = None
                    previous_serialized_observation = None
                if row['event']=='transition':
                    measure = args.planner_buffered_shadow or args.planner_async_shadow
                    if measure:
                        started = time.perf_counter()
                        row['policy_logits_seconds'] = (0. if choose.last.get('baseline_inference_skipped')
                                                        else logits_seconds)
                        row['previous_trace_timing'] = previous_trace_timing
                    for key in ('observation','next_observation'):
                        if measure and key=='observation' and previous_serialized_observation is not None:
                            row[key] = previous_serialized_observation
                        else:
                            row[key]={k:v.tolist() if hasattr(v,'tolist') else v for k,v in row[key].items()}
                    if measure:
                        previous_serialized_observation = row['next_observation']
                        annotated = choose.annotate(row)
                        previous_trace_timing = {'prepare_seconds': time.perf_counter() - started}
                        log.emit(annotated, console=False, timings=previous_trace_timing)
                        return
                log.emit(choose.annotate(row),console=row['event']!='transition')
            try:
                run_episodes(session,args.episodes,emit,trace=True,policy=policy,action_fn=choose)
            finally:
                if args.planner_async_shadow:
                    choose.close()
                log.emit(choose.summary())


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; keys released.',file=sys.stderr)
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'Error: {exc}',file=sys.stderr);sys.exit(1)
