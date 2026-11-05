"""Opt-in frozen-policy filtering by existing linear bullet-clearance geometry."""
import math

from env.action_emulator import DIRECTIONS
from .boundary_control import BoundaryCommitment

RISK_CONTRACT = "linear-clearance-gate-v1"
RISK_OPTIONS = {"prediction_frames": 12, "min_clearance": 4., "max_clearance_loss": 1.}


def canonical_risk_observation(obs):
    """Match the precision supplied to the neural policy, without mutating input."""
    import numpy as np
    return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}


def risk_candidates(obs, allowed):
    gaps = forecast_clearances(obs)
    best = max(gaps[a] for a in allowed)
    emergency = best < RISK_OPTIONS["min_clearance"]
    threshold = best if emergency else max(RISK_OPTIONS["min_clearance"], best - RISK_OPTIONS["max_clearance_loss"])
    return {"risk_allowed_actions": [a for a in allowed if gaps[a] >= threshold],
            "risk_clearances": gaps, "risk_best_clearance": best, "risk_threshold": threshold,
            "risk_candidate_mode": "emergency" if emergency else "safe"}


def forecast_clearances(obs):
    """Forecast all nine directions; physical endpoint validity is separate."""
    bullets = []
    for row in obs["bullets"]:
        if len(row) != 7 or not math.isfinite(float(row[6])):
            raise ValueError("Invalid bullet row")
        if row[6] <= 0:
            continue
        values = [float(v) for v in row[:5]]
        if not all(math.isfinite(v) for v in values) or values[4] < 0:
            raise ValueError("Invalid bullet geometry")
        bullets.append(values)
    gaps = []
    for keys in DIRECTIONS:
        dx = int("right" in keys) - int("left" in keys)
        dy = int("down" in keys) - int("up" in keys)
        speed = 2 / max(1., math.hypot(dx, dy))
        sx, sy = dx * speed, dy * speed
        gap = 24.
        for bx, by, vx, vy, radius in bullets:
            bx, by, vx, vy = bx * 384, by * 448, vx * 16 - sx, vy * 16 - sy
            speed2 = vx * vx + vy * vy
            closest = max(0., min(RISK_OPTIONS["prediction_frames"], -(bx * vx + by * vy) / speed2)) if speed2 else 0.
            gap = min(gap, math.hypot(bx + vx * closest, by + vy * closest) - radius * 32 - 1.25)
        gaps.append(gap)
    return gaps


class RiskBoundaryCommitment(BoundaryCommitment):
    """Filter fresh choices; only physical bounds interrupt an existing hold."""
    def __init__(self, *args, canonical_observation=False, **kwargs):
        if type(canonical_observation) is not bool:
            raise ValueError("Invalid risk observation mode")
        self.canonical_observation = canonical_observation
        self.risk_changes = self.risk_applied = 0
        super().__init__(*args, **kwargs)

    def _masked_action(self, obs):
        physical = super()._masked_action(obs)
        allowed = self._selection["boundary_allowed_actions"]
        details = risk_candidates(obs, allowed)
        candidates = details["risk_allowed_actions"]
        scores = self._selection["actor_logits"]
        selected = max(candidates, key=lambda a: (scores[a], -a))
        self._selection.update(boundary_student_action=physical,
                               masked_student_action=selected, risk_filtered_action=selected,
                               **details,
                               risk_mask_overridden=selected != physical)
        return selected

    def __call__(self, obs):
        if self.canonical_observation:
            obs = canonical_risk_observation(obs)
        action = super().__call__(obs)
        changed = self.last["risk_mask_overridden"]
        self.risk_changes += int(changed)
        self.risk_applied += int(changed and not self.last["commitment_active"])
        return action

    def summary(self):
        return {**super().summary(), "risk_contract": RISK_CONTRACT,
                "risk_changed_proposals": self.risk_changes,
                "risk_applied_fresh_choices": self.risk_applied}
