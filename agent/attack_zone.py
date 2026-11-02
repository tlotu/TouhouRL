"""Experimental rule ranking with a horizontal attack band and clearance gates."""
from dataclasses import dataclass
import math

from .baselines import action_candidates, attack_target


@dataclass(frozen=True)
class AttackZoneConfig:
    half_width: float = 24.0
    min_clearance: float = 4.0
    max_clearance_loss: float = 1.0

    def __post_init__(self):
        values = (self.half_width, self.min_clearance, self.max_clearance_loss)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Attack-zone parameters must be finite")
        if not 0 <= self.half_width <= 192 or not 0 <= self.min_clearance <= 24 or not 0 <= self.max_clearance_loss <= 24:
            raise ValueError("Attack-zone parameters are out of range")


def rank_attack_zone(obs, config):
    candidates = action_candidates(obs)
    target = attack_target(obs)
    tx, ty = (target[0] * 384, target[1] * 448) if target is not None else (0., 0.)
    if not candidates:
        return {"action": 0, "mode": "no_valid_direction", "target_dx": tx,
                "best_clearance": None, "threshold": None, "candidates": []}
    best = max(c[3] for c in candidates)
    threshold = max(config.min_clearance, best - config.max_clearance_loss)
    escape = best < config.min_clearance
    table = []
    for action, sx, sy, gap, score in candidates:
        excess = max(0., abs(tx - sx * 4) - config.half_width) if target is not None else 0.
        approach = max(0., ty - sy * 4 + 64) if target is not None else 0.
        table.append({"action": action, "clearance": gap, "reactive_score": score,
                      "eligible": gap >= threshold, "zone_excess": excess,
                      "approach_cost": approach})
    if escape:
        chosen = max(table, key=lambda c: (c["clearance"], c["reactive_score"], -c["action"]))
    else:
        chosen = max((c for c in table if c["eligible"]),
                     key=lambda c: (-c["zone_excess"] - 2*c["approach_cost"],
                                    c["reactive_score"], -c["action"]))
    return {"action": chosen["action"], "mode": "escape" if escape else
            "attack_band" if target is not None else "no_target", "target_dx": tx,
            "best_clearance": best, "threshold": threshold, "candidates": table}


class AttackZonePolicy:
    def __init__(self, config):
        self.config = config
        self.last = None

    def __call__(self, obs):
        self.last = rank_attack_zone(obs, self.config)
        return self.last["action"]
