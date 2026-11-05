"""Evaluate a frozen static-laser PPO checkpoint without training."""
import argparse
import hashlib
import math
from pathlib import Path
import sys

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
    args=parser.parse_args()
    if (args.task!=LASER_TASK or args.episodes<1 or not 0<args.focus_timeout<=60
            or not math.isfinite(args.candidate_slack) or not 0<=args.candidate_slack<=8):
        parser.error('Select the declared laser task and valid episode/focus settings')
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
    from env.windows import WindowsKeyboard,connect

    archive=Path(args.resume)/'model.zip'
    model_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    with DiagnosticLog(args.output) as log:
        reader,pid,build=connect(args.pid)
        try:
            if build!=saved['build_md5'] or 'SPELL_RESULT_ACTIVE' not in reader.addresses:
                raise RuntimeError('Laser evaluation requires the matching verified game build')
            reader=LaserReader(reader,build)
            controls=ActionEmulator(WindowsKeyboard(pid))
            session=LaserSession(reader,controls,repeat=2,capacity=config['capacity'],
                                 max_frames=config['max_frames'],spell_retry=True)
        except BaseException:
            reader.close();raise
        with session:
            learner=LaserPPOBackend(LaserTouhouEnv(session),checkpoint=str(archive))
            if type(learner.model.policy) is not LaserRiskPolicy or learner.timesteps!=saved['timesteps']:
                raise RuntimeError('Checkpoint policy or counters differ')
            learner.enable_laser(LaserTouhouEnv(session).observation_space,config['hold_frames'])
            choose=LaserEvaluationCommitment(learner.action_logits,lambda:session.elapsed_frames,
                                            config['hold_frames'],candidate_slack=args.candidate_slack)
            external_gate=args.candidate_slack!=config['risk_filter']['max_clearance_loss']
            policy='laser_ppo_external_slack_held' if external_gate else 'laser_ppo_deterministic_held'
            log.emit({'event':'connected','pid':pid,'build_md5':build,'schema':LASER_SCHEMA,
                'policy':policy,'collection':'laser_ppo_external_slack_evaluation' if external_gate else 'laser_ppo_evaluation','task':args.task,
                'task_verification':'manual','training':False,'checkpoint':str(Path(args.resume).resolve()),
                'model_sha256':model_hash,'timesteps':saved['timesteps'],'updates':saved['updates'],
                'hold_frames':config['hold_frames'],'repeat':2,'capacity':config['capacity'],
                'candidate_slack':args.candidate_slack,'external_candidate_gate':external_gate,
                'laser_filter':config['laser_filter'],'risk_filter':config['risk_filter'],
                'acquisition_contract':ACQUISITION_CONTRACT,'timing_clock':timing_clock_info()})
            print('Select the declared spell result menu and focus the game. Frozen deterministic evaluation.',file=sys.stderr)
            if external_gate:
                print(f'External evaluation gate: candidate slack {args.candidate_slack:g} px; '
                      'not the checkpoint sampling distribution and not PPO training.',file=sys.stderr)
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            def emit(row):
                if row['event']=='transition':
                    for key in ('observation','next_observation'):
                        row[key]={k:v.tolist() if hasattr(v,'tolist') else v for k,v in row[key].items()}
                log.emit(choose.annotate(row),console=row['event']!='transition')
            try:
                run_episodes(session,args.episodes,emit,trace=True,policy=policy,action_fn=choose)
            finally:
                log.emit(choose.summary())


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; keys released.',file=sys.stderr)
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'Error: {exc}',file=sys.stderr);sys.exit(1)
