"""Train one static-laser task with a versioned, risk-consistent PPO policy."""
import argparse
from functools import partial
import hashlib
from pathlib import Path
import sys
import time

from agent.checkpoints import create_run_directory,read_checkpoint,save_checkpoint
from agent.laser_contract import LASER_PPO_CONTRACT,LASER_SCHEMA,LASER_TASK
from agent.laser_policy import LASER_CAPACITY,LASER_OBSERVATION_CONTRACT,LaserRiskPolicy
from agent.masked_contract import MASKED_REWARD_PROFILE
from agent.risk_contract import RISK_PPO_CONTRACT
from scripts.diagnostics import DiagnosticLog,timing_clock_info

LASER_CONFIG={'observation_contract':LASER_OBSERVATION_CONTRACT,'capacity':LASER_CAPACITY,
              'geometry':'static-capsule-integer-12-v1','phase':'reference-timing-user-observed-v1',
              'player_radius':1.25,'source_task':LASER_TASK}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--decisions',type=int,default=256)
    parser.add_argument('--rollout-decisions',type=int,default=256)
    parser.add_argument('--pid',type=int)
    parser.add_argument('--focus-timeout',type=float,default=30.)
    args=parser.parse_args()
    if args.task!=LASER_TASK:
        parser.error('This experimental contract is restricted to the manually selected static-laser task')
    if not 2<=args.decisions<=1000000 or not 2<=args.rollout_decisions<=65536 or not 0<args.focus_timeout<=60:
        parser.error('Invalid decision budget or focus timeout')
    saved=read_checkpoint(args.resume)
    oldconfig=saved['config'];contract=oldconfig.get('ppo_contract')
    if contract==RISK_PPO_CONTRACT:
        from scripts.train_masked import risk_config
        try: config=risk_config(saved)
        except ValueError as exc:parser.error(str(exc))
        config={**config,'ppo_contract':LASER_PPO_CONTRACT,'laser_filter':LASER_CONFIG}
        migrating=True
    elif contract==LASER_PPO_CONTRACT:
        config=oldconfig
        if config.get('laser_filter')!=LASER_CONFIG or saved.get('reward_profile')!=MASKED_REWARD_PROFILE:
            parser.error('Saved laser contract differs')
        from scripts.train_masked import risk_config
        prior={**saved,'config':{**config,'ppo_contract':RISK_PPO_CONTRACT}}
        prior['config'].pop('laser_filter')
        try: risk_config(prior)
        except ValueError as exc:parser.error(str(exc))
        migrating=False
    else:
        parser.error('Source must be a risk PPO checkpoint or matching laser checkpoint')
    from agent.laser_backend import LaserPPOBackend
    from agent.masked_training import collect_masked_episodes
    from agent.risk_policy import RiskActorCriticPolicy
    from agent.training import train_episodic
    from env.action_emulator import ActionEmulator
    from env.frame_stepper import ACQUISITION_CONTRACT
    from env.laser_session import LaserReader,LaserSession,LaserTouhouEnv,laser_state_space
    from env.touhou_env import state_space
    from env.windows import WindowsKeyboard,connect
    import gymnasium as gym

    class Offline(gym.Env):
        action_space=gym.spaces.Discrete(9)
        def __init__(self,space):self.observation_space=space
        def reset(self,**kwargs):raise RuntimeError('Offline preparation cannot reset a game')
        def step(self,action):raise RuntimeError('Offline preparation cannot send input')

    source=Path(args.resume)/'model.zip'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    newspace=laser_state_space(config['capacity'])
    learner=LaserPPOBackend(Offline(state_space(config['capacity']) if migrating else newspace),checkpoint=str(source))
    if migrating and type(learner.model.policy) is not RiskActorCriticPolicy:
        raise RuntimeError('Source policy is not the expected risk policy')
    if not migrating and type(learner.model.policy) is not LaserRiskPolicy:
        raise RuntimeError('Source policy is not the expected laser policy')
    if learner.timesteps!=saved['timesteps']:
        raise RuntimeError('Source checkpoint counters differ')
    learner.enable_laser(newspace,config['hold_frames'])
    learner.configure(config['held_ppo'])
    lineage={'source_checkpoint':str(source.parent.resolve()),'source_model_sha256':source_hash,
             'task':args.task,'method':LASER_PPO_CONTRACT,'prior':saved.get('fine_tuning')}
    run=create_run_directory(args.run_dir)
    def publish(state):
        return save_checkpoint(learner,run,{**state,'schema':LASER_SCHEMA,
            'reward_profile':MASKED_REWARD_PROFILE,'initialization':saved['initialization'],
            'fine_tuning':lineage},config,saved['build_md5'])
    with DiagnosticLog(run/'training.jsonl') as log:
        log.emit({'event':'laser_preparation','task':args.task,'config':config,
                  'source_model_sha256':source_hash,'prepare_only':args.prepare_only,
                  'acquisition_contract':ACQUISITION_CONTRACT,
                  'laser_semantics':'reference-phase-user-observed-static-only'})
        if args.prepare_only:
            checkpoint=publish({'updates':saved['updates'],'timesteps':learner.timesteps})
            log.emit({'event':'prepared_checkpoint','checkpoint':str(checkpoint)})
            return
        reader,pid,build=connect(args.pid)
        try:
            if build!=saved['build_md5'] or 'SPELL_RESULT_ACTIVE' not in reader.addresses:
                raise RuntimeError('Laser training requires the matching verified game build')
            reader=LaserReader(reader,build)
            controls=ActionEmulator(WindowsKeyboard(pid))
            session=LaserSession(reader,controls,repeat=2,capacity=config['capacity'],
                                 max_frames=config['max_frames'],spell_retry=True)
        except BaseException:
            reader.close()
            raise
        with session:
            controls.wait_for_focus(args.focus_timeout)
            session.verify_training_boundary()
            initial=publish({'updates':saved['updates'],'timesteps':learner.timesteps})
            log.emit({'event':'training_start','task':args.task,'checkpoint':str(initial),
                      'additional_decisions':args.decisions,'rollout_decisions':args.rollout_decisions,
                      'reward_profile':MASKED_REWARD_PROFILE,'fine_tuning':lineage,
                      'timing_clock':timing_clock_info()})
            collector=partial(collect_masked_episodes,hold_frames=config['hold_frames'],
                              gamma_frame=config['held_ppo']['gamma_frame'])
            train_episodic(LaserTouhouEnv(session),learner,session.verify_training_boundary,
                publish,log.emit,args.decisions,args.rollout_decisions,saved['updates'],
                clock=time.perf_counter,collector=collector)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Stopped; keys released. Resume only a complete checkpoint.',file=sys.stderr)
    except (OSError,RuntimeError,ValueError) as exc:
        print(f'Error: {exc}',file=sys.stderr);sys.exit(1)
