"""Risk-policy migration, on-policy updates, restoration and native-entry wiring."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.test_ppo_backend import AVAILABLE
from tests import test_masked_ppo as masked_tests
from tests.test_training import FakeTask
from tests.fakes import FakeKeyboard
from agent.held_training import HeldTransition
from agent.training import TrainingError


@unittest.skipUnless(AVAILABLE, "Install training dependencies")
class RiskPPOTests(unittest.TestCase):
    def make(self):
        from agent.risk_backend import RiskPPOBackend
        from scripts.train_masked import risk_config
        from agent.masked_contract import MASKED_REWARD_PROFILE
        learner, config, obs, Env = masked_tests.MaskedPPOTests().make()
        learner.enable_mask(4); learner.configure(config['held_ppo'])
        saved = {'config': config, 'timesteps': 0, 'updates': 0, 'reward_profile': MASKED_REWARD_PROFILE}
        return learner, risk_config(saved), obs, Env

    def convert(self, learner):
        from agent.risk_backend import RiskPPOBackend
        learner.__class__ = RiskPPOBackend
        learner.enable_mask(4)
        return learner

    def batch(self, learner, obs):
        result=[]
        for i in range(8):
            d=learner.act(obs)
            result.append(HeldTransition(deepcopy(obs), d.action, .1+i*.02, i==0, d.value, d.log_prob, 4))
        return result

    def test_configuration_migration_and_resume_guards(self):
        from scripts.train_masked import risk_config, masked_config
        from agent.masked_contract import MASKED_REWARD_PROFILE
        learner,config,obs,Env=self.make()
        saved={'config':config,'timesteps':8,'updates':1,'reward_profile':MASKED_REWARD_PROFILE}
        self.assertEqual(risk_config(saved),config)
        with self.assertRaises(ValueError):masked_config(saved)
        broken=deepcopy(saved);broken['config']['risk_filter']['observation_contract']='unknown'
        with self.assertRaises(ValueError):risk_config(broken)

    def test_controller_mask_and_sampled_likelihood_agree(self):
        import torch
        from agent.risk_control import RiskBoundaryCommitment
        learner,config,obs,Env=self.make();learner=self.convert(learner)
        obs['bullets'][0]=[0,12/448,0,0,2/32,.25,1]
        for x,y in ((192,384),(374,300),(8,432),(366,384)):
            obs['player'][:2]=[x/192-1,y/224-1]
            tensor,_=learner.model.policy.obs_to_tensor(obs)
            mask=learner.model.policy.action_mask(tensor)[0].tolist()
            controller=RiskBoundaryCommitment(learner.action_logits,lambda:0,canonical_observation=True)
            action=controller(obs);row=controller.annotate({'event':'transition','action':action})
            self.assertEqual([a for a,v in enumerate(mask)if v],row['risk_allowed_actions'])
            self.assertEqual(action,learner.deterministic_action(obs))
            for _ in range(4):
                d=learner.act(obs);self.assertTrue(mask[d.action])
                _,lp,entropy=learner.model.policy.evaluate_actions(tensor,torch.tensor([d.action]))
                self.assertAlmostEqual(d.log_prob,float(lp[0].detach()),places=6)
                self.assertTrue(torch.isfinite(entropy).all())

    def test_conversion_preserves_parameters_and_nonempty_adam(self):
        import torch
        learner,config,obs,Env=self.make()
        learner.update(self.batch(learner,obs),16)
        state=deepcopy(learner.model.policy.state_dict());optim=deepcopy(learner.model.policy.optimizer.state_dict())
        learner=self.convert(learner)
        for k,v in state.items():self.assertTrue(torch.equal(v,learner.model.policy.state_dict()[k]))
        restored=learner.model.policy.optimizer.state_dict()
        self.assertEqual(optim['param_groups'],restored['param_groups'])
        for k,values in optim['state'].items():
            for key,v in values.items():self.assertTrue(torch.equal(v,restored['state'][k][key]))
        self.assertEqual(learner.timesteps,8)

    def test_actual_actor_update_reload_and_bad_likelihood_rejection(self):
        import torch
        from dataclasses import replace
        from agent.risk_backend import RiskPPOBackend
        learner,config,obs,Env=self.make();learner=self.convert(learner)
        obs['bullets'][0]=[0,12/448,0,0,2/32,.25,1]
        rows=self.batch(learner,obs)
        with self.assertRaises(TrainingError):learner.update([replace(r,log_prob=r.log_prob+1)for r in rows],16)
        self.assertEqual(learner.timesteps,0)
        before=deepcopy(learner.model.policy.state_dict());learner.update(rows,16)
        self.assertTrue(any(k.startswith(('action_net.','mlp_extractor.policy_net.'))and not torch.equal(v,learner.model.policy.state_dict()[k])for k,v in before.items()))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            p=Path(directory)/'model.zip';learner.save(p)
            restored=RiskPPOBackend(Env(),checkpoint=str(p));restored.enable_mask(4);restored.configure(config['held_ppo'])
            for k,values in learner.model.policy.optimizer.state_dict()['state'].items():
                for key,v in values.items():self.assertTrue(torch.equal(v,restored.model.policy.optimizer.state_dict()['state'][k][key]))
            restored.update(self.batch(restored,obs),24)
            self.assertEqual(restored.timesteps,16)

    def source(self, root):
        from agent.checkpoints import save_checkpoint
        learner,config,obs,Env=self.make()
        physical=dict(config);physical.pop('risk_filter');physical['ppo_contract']='masked-held-smdp-v1'
        source=save_checkpoint(learner,root,{'updates':0,'timesteps':0,'reward_profile':'spell-v3-masked-survival',
          'initialization':{'identity':{'tasks':['a','b']}}},physical,'test')
        return source,obs,Env

    def test_prepare_and_evaluation_guards_preserve_source(self):
        from scripts.train_masked import main
        from scripts.evaluate_control import main as evaluate
        from agent.checkpoints import read_checkpoint
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            source,obs,Env=self.source(Path(directory)/'source');before=(source/'model.zip').read_bytes()
            args=['train_masked','--risk-filter','--prepare-only','--resume',str(source),'--task','a','--run-dir',str(Path(directory)/'prepared')]
            with patch('sys.argv',args),patch('env.windows.connect')as connect,contextlib.redirect_stdout(io.StringIO()):main()
            connect.assert_not_called()
            checkpoint=Path(directory)/'prepared/checkpoints/update-000000';read_checkpoint(checkpoint)
            args=['evaluate','--boundary-mask','--resume',str(checkpoint),'--task','a','--output','unused']
            with patch('sys.argv',args),patch('scripts.evaluate_control.connect')as connect,contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):evaluate()
            connect.assert_not_called();self.assertEqual((source/'model.zip').read_bytes(),before)
            class Task(FakeTask):
                capacity=4
                @property
                def elapsed_frames(self):return self.position*2
                def reset(self):
                    _,info=super().reset();return {k:v.tolist()for k,v in obs.items()},info
                def step(self,action):
                    _,reward,done,truncated,info=super().step(action)
                    return {k:v.tolist()for k,v in obs.items()},reward,done,truncated,info
                def __enter__(self):return self
                def __exit__(self,*args):return False
                def verify_training_boundary(self):self.boundary()
            output=Path(directory)/'eval.jsonl';args[args.index('unused')]=str(output);args.append('--risk-filter')
            reader=SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE':1},close=lambda:None)
            with patch('sys.argv',args),patch('scripts.evaluate_control.connect',return_value=(reader,1,'test')), \
                    patch('scripts.evaluate_control.WindowsKeyboard',return_value=FakeKeyboard()), \
                    patch('scripts.evaluate_control.GameSession',return_value=Task(length=4)), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):evaluate()
            header=json.loads(output.read_text().splitlines()[0])
            self.assertEqual(header['risk_observation_contract'],'float32-v1')
            self.assertEqual((source/'model.zip').read_bytes(),before)

    def test_fake_native_risk_training_publishes_actual_update(self):
        from scripts.train_masked import main
        from agent.checkpoints import read_checkpoint
        from agent.risk_backend import RiskPPOBackend
        learner,config,state,Env=self.make()
        class Task(FakeTask):
            capacity=4
            def reset(self):
                _,info=super().reset();return deepcopy(state),info
            def step(self,action):
                _,reward,done,truncated,info=super().step(action);info['deaths']=0
                return deepcopy(state),reward,done,truncated,info
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def verify_training_boundary(self):self.boundary()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            source,obs,Env=self.source(Path(directory)/'source');before=(source/'model.zip').read_bytes();run=Path(directory)/'run'
            args=['train_masked','--risk-filter','--resume',str(source),'--task','a','--decisions','2','--rollout-decisions','2','--run-dir',str(run)]
            reader=SimpleNamespace(addresses={'SPELL_RESULT_ACTIVE':1},close=lambda:None)
            task=Task(length=4)
            with patch('sys.argv',args),patch('env.windows.connect',return_value=(reader,1,'test')),patch('env.windows.WindowsKeyboard',return_value=FakeKeyboard()),patch('env.session.GameSession',return_value=task),contextlib.redirect_stdout(io.StringIO()):main()
            cp=run/'checkpoints/update-000001';saved=read_checkpoint(cp)
            restored=RiskPPOBackend(Env(),checkpoint=str(cp/'model.zip'))
            self.assertEqual((saved['updates'],restored.timesteps),(1,2))
            self.assertEqual(saved['config']['ppo_contract'],'risk-masked-held-smdp-v1')
            self.assertEqual((source/'model.zip').read_bytes(),before)
            self.assertTrue(task.menu)
            rows=[json.loads(s)for s in (run/'training.jsonl').read_text().splitlines()]
            held=[r for r in rows if r['event']=='held_decision']
            self.assertTrue(held)
            self.assertTrue(all(r['action']in r['sampling_context']['risk_allowed_actions']for r in held))
