"""Evaluation-only attack-band ranking inside the unchanged risk candidate set."""
import math

from env.action_emulator import DIRECTIONS
from .risk_control import RiskBoundaryCommitment

ATTACK_RETURN_CONTRACT = "safe-attack-band-return-v1"


def return_choice(obs, candidates, scores, mode, hold_frames, half_width):
    """Prefer horizontal recovery, vertical detours, stay, then retreat."""
    base = max(candidates, key=lambda a: (scores[a], -a))
    eligible = [r for r in obs['enemies'] if len(r) == 6
                and all(math.isfinite(float(v)) for v in r)
                and r[5] > 0 and r[3] > 0 and r[4] > 0 and r[1] < 0]
    target = min(eligible, key=lambda r: (-float(r[2]), abs(float(r[0])), abs(float(r[1])))) if eligible else None
    dx = float(target[0]) * 384 if target is not None else None
    excess = max(0., abs(dx) - half_width) if dx is not None else 0.
    details = {'attack_return_target_dx': dx, 'attack_return_zone_excess': excess,
               'attack_return_active': bool(excess > 1e-4 and mode == 'safe'),
               'attack_return_base_action': base}
    if not details['attack_return_active']:
        return base, details

    def rank(action):
        keys = DIRECTIONS[action]
        vx = int('right' in keys) - int('left' in keys)
        vy = int('down' in keys) - int('up' in keys)
        motion = vx * 2 / max(1., math.hypot(vx, vy)) * hold_frames
        remaining = max(0., abs(dx - motion) - half_width)
        group = 0 if remaining < excess - 1e-4 else 1 if vx == 0 and vy else 2 if action == 0 else 3
        return (-group, scores[action], -action)

    return max(candidates, key=rank), details


class AttackReturnCommitment(RiskBoundaryCommitment):
    def __init__(self, *args, half_width=24., **kwargs):
        if not math.isfinite(half_width) or not 0 <= half_width <= 192:
            raise ValueError('Invalid attack return half-width')
        self.half_width = half_width
        self.return_changes = self.return_applied = 0
        super().__init__(*args, **kwargs)

    def _masked_action(self, obs):
        base = super()._masked_action(obs)
        selection = self._selection
        action, details = return_choice(obs, selection['risk_allowed_actions'],
            selection['actor_logits'], selection['risk_candidate_mode'], self.hold_frames, self.half_width)
        selection.update(details, masked_student_action=action,
            attack_return_action=action, attack_return_overridden=action != base)
        return action

    def __call__(self, obs):
        action = super().__call__(obs)
        changed = self.last['attack_return_overridden']
        self.return_changes += int(changed)
        self.return_applied += int(changed and not self.last['commitment_active'])
        return action

    def summary(self):
        return {**super().summary(), 'attack_return_contract': ATTACK_RETURN_CONTRACT,
                'attack_return_changed_proposals': self.return_changes,
                'attack_return_applied_fresh_choices': self.return_applied}
