"""Experimental PPO backend for in-memory laser policy migration and updates."""
import numpy as np
from .laser_policy import LaserHazardFeatures, LaserRiskPolicy, decode_lasers, migrate_risk_policy
from .laser_risk import combined_candidates
from .masked_backend import MaskedPPOBackend


class LaserPPOBackend(MaskedPPOBackend):
    def enable_laser(self, observation_space, hold_frames):
        if isinstance(self.model.policy,LaserRiskPolicy):
            if self.model.policy.mask_hold_frames!=hold_frames or self.model.observation_space!=observation_space:
                raise ValueError('Saved laser policy settings differ')
            return
        source=self.model.policy
        if getattr(source,'mask_hold_frames',None)!=hold_frames:
            raise ValueError('Source hold horizon differs')
        policy=migrate_risk_policy(source,observation_space,self.model.lr_schedule,self.model.policy_kwargs)
        kwargs=dict(self.model.policy_kwargs)
        kwargs['features_extractor_class']=LaserHazardFeatures
        self.model.policy,self.model.policy_class,self.model.policy_kwargs=policy,LaserRiskPolicy,kwargs
        self.model.observation_space=observation_space

    def act(self,obs,deterministic=False):
        decision=super().act(obs,deterministic=deterministic)
        self.decision_context={}
        if isinstance(self.model.policy,LaserRiskPolicy):
            details=combined_candidates(obs,decode_lasers(obs['lasers']),self.model.policy.mask_hold_frames)
            if decision.action not in details['candidates']:
                raise RuntimeError('Sampled action differs from combined laser risk candidates')
            self.decision_context={**details,'laser_count':int(sum(row[11]>0 for row in obs['lasers'])),
                'player':np.asarray(obs['player'],dtype=np.float32).tolist(),
                'laser_rows':np.asarray(obs['lasers'],dtype=np.float32).tolist()}
        return decision
