"""Offline preparation and fake-native menu-boundary laser PPO training."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.checkpoints import read_checkpoint,save_checkpoint
from agent.laser_contract import LASER_PPO_CONTRACT,LASER_SCHEMA,LASER_TASK
from agent.laser_policy import encode_lasers
from agent.masked_contract import MASKED_REWARD_PROFILE
from tests.fakes import FakeKeyboard
from tests.test_laser_geometry import laser
from tests import test_risk_ppo
from tests.test_training import FakeTask


class LaserTrainingTests(unittest.TestCase):
    def make_source(self,root):
        helper=test_risk_ppo.RiskPPOTests();learner,config,obs,Env=helper.make();helper.convert(learner)
        source=save_checkpoint(learner,root,{'updates':0,'timesteps':0,
            'reward_profile':MASKED_REWARD_PROFILE,
            'initialization':{'identity':{'tasks':['a','b']}}},config,'test-build')
        return source,obs

    def test_offline_prepare_and_wrong_task_guard(self):
        from scripts.train_laser import main
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            root=Path(directory);source,_=self.make_source(root/'source')
            before=(source/'model.zip').read_bytes();run=root/'prepared'
            args=['train_laser','--resume',str(source),'--task',LASER_TASK,'--run-dir',str(run),'--prepare-only']
            with patch('sys.argv',args),patch('env.windows.connect')as connect, \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main()
            connect.assert_not_called()
            saved=read_checkpoint(run/'checkpoints/update-000000')
            self.assertEqual(saved['schema'],LASER_SCHEMA)
            self.assertEqual(saved['config']['ppo_contract'],LASER_PPO_CONTRACT)
            self.assertEqual((source/'model.zip').read_bytes(),before)
            args[args.index(LASER_TASK)]='unrelated-task'
            with patch('sys.argv',args),patch('env.windows.connect')as connect, \
                    contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
            connect.assert_not_called()

    def test_fake_native_full_episode_updates_at_result_menu(self):
        from scripts.train_laser import main
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            root=Path(directory);source,obs=self.make_source(root/'source');before=(source/'model.zip').read_bytes()
            prep=root/'preflight'
            prep_args=['train_laser','--resume',str(source),'--task',LASER_TASK,'--run-dir',str(prep),'--prepare-only']
            with patch('sys.argv',prep_args),patch('env.windows.connect')as connect, \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main()
            connect.assert_not_called()
            prepared=prep/'checkpoints/update-000000'
            obs={k:v.tolist()for k,v in obs.items()}
            beam=laser();beam['segment_candidate']=[[100.,200.],[280.,200.]]
            obs['lasers']=encode_lasers([beam]).tolist()
            class Task(FakeTask):
                capacity=4
                def reset(self):
                    _,info=super().reset();return deepcopy(obs),info
                def step(self,action):
                    _,reward,done,truncated,info=super().step(action);info['deaths']=0
                    return deepcopy(obs),reward,done,truncated,info
                def __enter__(self):return self
                def __exit__(self,*args):return False
                def verify_training_boundary(self):self.boundary()
            task=Task(length=4);run=root/'trained'
            args=['train_laser','--resume',str(prepared),'--task',LASER_TASK,'--run-dir',str(run),
                  '--decisions','2','--rollout-decisions','2']
            reader=SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE':1},close=lambda:None)
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test-build')), \
                    patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('env.laser_session.LaserReader',side_effect=lambda r,b:r), \
                    patch('env.laser_session.LaserSession',return_value=task), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main()
            saved=read_checkpoint(run/'checkpoints/update-000001')
            self.assertEqual((saved['updates'],saved['timesteps']),(1,2))
            self.assertTrue(task.menu)
            self.assertEqual((source/'model.zip').read_bytes(),before)
            rows=[json.loads(l)for l in (run/'training.jsonl').read_text().splitlines()]
            held=[r for r in rows if r['event']=='held_decision']
            self.assertEqual(len(held),2)
            self.assertTrue(all(r['action'] in r['sampling_context']['candidates'] for r in held))
            self.assertTrue(all(r['sampling_context']['laser_count']==1 for r in held))
            from agent.laser_geometry import forecast_static_lasers
            from agent.laser_policy import decode_lasers
            for row in held:
                ctx=row['sampling_context'];player=ctx['player']
                point=((player[0]+1)*192,(player[1]+1)*224)
                self.assertEqual(ctx['laser_clearances'],forecast_static_lasers(point,decode_lasers(ctx['laser_rows'])))
