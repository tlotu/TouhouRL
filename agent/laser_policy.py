"""Experimental offline laser-aware PPO policy; no live training integration."""
from copy import deepcopy
import numpy as np
import torch

from .hazard_features import HazardFeatures
from .laser_geometry import forecast_static_lasers
from .laser_risk import combined_candidates
from .masked_policy import BoundaryActorCriticPolicy

LASER_OBSERVATION_CONTRACT = 'static-laser-rows-f32-v1'
LASER_CAPACITY = 64
LASER_FIELDS = 12


def encode_lasers(lasers):
    """Encode candidate segments and raw phase/timing; reject truncation."""
    if len(lasers)>LASER_CAPACITY:
        raise ValueError('Laser capacity exceeded')
    output=np.zeros((LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)
    for i,l in enumerate(lasers):
        if l['issues'] or l['segment_candidate'] is None:
            raise ValueError('Invalid laser candidate')
        (ax,ay),(bx,by)=l['segment_candidate']
        output[i]=[ax,ay,bx,by,l['width'],l['phase_raw'],l['timer_raw'],
                   l['warning_frames_raw'],l['firing_frames_raw'],l['fading_frames_raw'],l['speed'],1]
    decode_lasers(output)
    return output


def decode_lasers(rows):
    if np.asarray(rows).shape != (LASER_CAPACITY,LASER_FIELDS):
        raise ValueError('Invalid laser observation shape')
    result=[]
    for row in rows:
        if not np.isfinite(row).all() or row[11] not in (0,1):
            raise ValueError('Invalid laser observation')
        if row[11]==0:
            continue
        phase,timer,warn,fire,fade=map(float,row[5:10])
        if any(v!=int(v) or v<0 for v in (phase,timer,warn,fire,fade)) or phase not in (0,1,2):
            raise ValueError('Invalid laser phase or timing')
        if row[4]<0 or row[10]!=0:
            raise ValueError('Invalid or moving laser')
        result.append({'segment_candidate':[[float(row[0]),float(row[1])],[float(row[2]),float(row[3])]],
            'width':float(row[4]),'phase_raw':int(phase),'timer_raw':int(timer),
            'warning_frames_raw':int(warn),'firing_frames_raw':int(fire),
            'fading_frames_raw':int(fade),'speed':float(row[10]),'issues':[]})
    return result


class LaserHazardFeatures(HazardFeatures):
    def __init__(self, observation_space, half_width=24.):
        super().__init__(observation_space,half_width=half_width)
        self._features_dim=71

    def forward(self, obs):
        base=super().forward(obs)
        states={k:v.detach().cpu().float().numpy() for k,v in obs.items()}
        values=[]
        for player,rows in zip(states['player'],states['lasers']):
            point=((float(player[0])+1)*192,(float(player[1])+1)*224)
            values.append(forecast_static_lasers(point,decode_lasers(rows)))
        gaps=torch.as_tensor(np.asarray(values),device=base.device,dtype=base.dtype)
        return torch.cat((base,(gaps/24).clamp(-2,1)),dim=1)


class LaserRiskPolicy(BoundaryActorCriticPolicy):
    """Apply combined candidates in the sampled and reevaluated distribution."""
    def __init__(self,*args,**kwargs):
        kwargs.setdefault('features_extractor_class',LaserHazardFeatures)
        super().__init__(*args,**kwargs)
        if not isinstance(self.features_extractor,LaserHazardFeatures):
            raise ValueError('Laser policy requires laser-aware features')

    def action_mask(self,obs):
        arrays={k:v.detach().cpu().float().numpy()for k,v in obs.items()}
        masks=[]
        for i in range(len(arrays['player'])):
            state={k:v[i]for k,v in arrays.items()}
            candidates=combined_candidates(state,decode_lasers(state['lasers']),self.mask_hold_frames)['candidates']
            masks.append([a in candidates for a in range(9)])
        return torch.tensor(masks,dtype=torch.bool,device=obs['player'].device)


def migrate_risk_policy(source, observation_space, lr_schedule, policy_kwargs):
    """Copy a compatible risk policy and Adam state into 71-feature input."""
    from .risk_policy import RiskActorCriticPolicy
    if type(source) is not RiskActorCriticPolicy:
        raise ValueError('Requires an intrinsic risk PPO policy')
    kwargs=dict(policy_kwargs)
    kwargs['features_extractor_class']=LaserHazardFeatures
    target=LaserRiskPolicy(observation_space,source.action_space,lr_schedule,**kwargs).to(source.device)
    old=source.state_dict();new=target.state_dict()
    if old.keys()!=new.keys():
        raise ValueError('Incompatible policy parameters')
    expanded={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
    for name,value in old.items():
        if value.shape==new[name].shape:
            new[name]=value.detach().clone()
        elif name in expanded and value.ndim==2 and value.shape[1]==62 and new[name].shape==(value.shape[0],71):
            replacement=torch.zeros_like(new[name]);replacement[:,:62]=value
            new[name]=replacement
        else:
            raise ValueError(f'Incompatible parameter shape: {name}')
    target.load_state_dict(new,strict=True)
    oldparams=dict(source.named_parameters());newparams=dict(target.named_parameters())
    if oldparams.keys()!=newparams.keys() or len(source.optimizer.param_groups)!=1 or len(target.optimizer.param_groups)!=1:
        raise ValueError('Incompatible optimizer structure')
    for name,oldparam in oldparams.items():
        if oldparam not in source.optimizer.state:
            continue
        newparam=newparams[name];newstate={}
        for key,value in source.optimizer.state[oldparam].items():
            if not torch.is_tensor(value):
                newstate[key]=deepcopy(value)
            elif value.ndim==0:
                newstate[key]=value.detach().clone()
            elif value.shape==newparam.shape:
                newstate[key]=value.detach().clone()
            elif name in expanded and value.shape==oldparam.shape and newparam.shape[1]==71:
                replacement=torch.zeros_like(newparam);replacement[:,:62]=value
                newstate[key]=replacement
            else:
                raise ValueError(f'Incompatible optimizer state: {name}/{key}')
        target.optimizer.state[newparam]=newstate
    for key,value in source.optimizer.param_groups[0].items():
        if key!='params':target.optimizer.param_groups[0][key]=deepcopy(value)
    target.set_training_mode(False)
    return target
