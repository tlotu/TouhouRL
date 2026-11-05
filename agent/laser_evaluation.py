"""Deterministic held evaluation of a frozen combined-risk laser policy."""
import math
from .boundary_control import BoundaryCommitment
from .laser_policy import decode_lasers
from .laser_risk import combined_candidates
from .risk_control import canonical_risk_observation


class LaserEvaluationCommitment(BoundaryCommitment):
    def __init__(self,*args,candidate_slack=1.,**kwargs):
        if not math.isfinite(candidate_slack) or not 0 <= candidate_slack <= 8:
            raise ValueError('Candidate slack must be between 0 and 8 pixels')
        self.candidate_slack=float(candidate_slack)
        self.risk_changes=self.risk_applied=0
        super().__init__(*args,**kwargs)

    def _masked_action(self,obs):
        physical=super()._masked_action(obs)
        detail=combined_candidates(obs,decode_lasers(obs['lasers']),self.hold_frames)
        if self.candidate_slack!=1. and detail['mode']=='safe':
            detail['threshold']=max(4.,detail['best']-self.candidate_slack)
            detail['candidates']=[a for a in detail['allowed']
                                  if detail['combined_clearances'][a]>=detail['threshold']]
        scores=self._selection['actor_logits']
        action=max(detail['candidates'],key=lambda a:(scores[a],-a))
        self._selection.update(boundary_student_action=physical,masked_student_action=action,
            laser_risk_filtered_action=action,laser_risk_overridden=action!=physical,
            laser_risk_context=detail)
        return action

    def __call__(self,obs):
        action=super().__call__(canonical_risk_observation(obs))
        changed=self.last['laser_risk_overridden']
        self.risk_changes+=int(changed)
        self.risk_applied+=int(changed and not self.last['commitment_active'])
        return action

    def summary(self):
        return {**super().summary(),'laser_risk_changed_proposals':self.risk_changes,
                'laser_risk_applied_fresh_choices':self.risk_applied}
