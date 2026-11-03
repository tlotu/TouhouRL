"""Optional evaluation-only endpoint masking for the actual action hold."""
import math
import time

from .action_commitment import ActionCommitment
from env.action_emulator import DIRECTIONS


BOUNDARY_CONTRACT = "hold-endpoint-mask-v1"
FIELD_BOUNDS = {"x_min": 8., "x_max": 376., "y_min": 16., "y_max": 432.}


def feasible_actions(obs, frames):
    """Reject nominal motion crossing the existing field bounds within frames."""
    if type(frames) is not int or not 1 <= frames <= 16:
        raise ValueError("Boundary horizon must be an integer from 1 to 16")
    px, py = (float(obs["player"][0]) + 1) * 192, (float(obs["player"][1]) + 1) * 224
    bounds, epsilon = FIELD_BOUNDS, 1e-4
    def inside(x, y):
        return (bounds["x_min"] - epsilon <= x <= bounds["x_max"] + epsilon
                and bounds["y_min"] - epsilon <= y <= bounds["y_max"] + epsilon)
    if not math.isfinite(px) or not math.isfinite(py) or not inside(px, py):
        raise ValueError("Player position is outside the supported boundary model")
    allowed = []
    for action, keys in enumerate(DIRECTIONS):
        dx = int("right" in keys) - int("left" in keys)
        dy = int("down" in keys) - int("up" in keys)
        speed = 2 / max(1., math.hypot(dx, dy))
        if inside(px + dx * speed * frames, py + dy * speed * frames):
            allowed.append(action)
    return allowed


class BoundaryCommitment(ActionCommitment):
    """Hold masked actor choices, interrupting only for projected boundary crossing."""
    def __init__(self, logits, elapsed_frames, hold_frames=4, clock=time.perf_counter):
        self.logits = logits
        self.mask_overrides = self.total_overrides = self.interruptions = 0
        super().__init__(self._masked_action, elapsed_frames, hold_frames, clock)

    def _masked_action(self, obs):
        scores = [float(value) for value in self.logits(obs)]
        if len(scores) != 9 or not all(math.isfinite(value) for value in scores):
            raise ValueError("Boundary selection requires nine finite actor logits")
        allowed = feasible_actions(obs, self.hold_frames)
        raw = max(range(9), key=lambda a: (scores[a], -a))
        selected = max(allowed, key=lambda a: (scores[a], -a))
        self._selection = {"raw_student_action": raw, "masked_student_action": selected,
                           "boundary_allowed_actions": allowed, "actor_logits": scores,
                           "boundary_mask_overridden": raw != selected}
        return selected

    def __call__(self, obs):
        if self.last is not None:
            raise RuntimeError("Missing transition annotation")
        frame = self.elapsed_frames()
        age = frame - self.since if self.since is not None else None
        active = age is not None and 0 <= age < self.hold_frames
        # Next substep requests two frames even if the hold has only one left.
        remaining = max(2, self.hold_frames - age) if active else self.hold_frames
        interrupted = active and self.held_action not in feasible_actions(obs, remaining)
        previous_age = age
        if interrupted:
            self.since = None
        action = super().__call__(obs)
        holding_override = self.last["action_overridden"]
        execution_horizon = remaining if self.last["commitment_active"] else self.hold_frames
        execution_allowed = feasible_actions(obs, execution_horizon)
        if action not in execution_allowed:
            raise RuntimeError("Boundary controller selected an infeasible held action")
        self.last.update(self._selection)
        self.last.update(commitment_overridden=holding_override,
                         action_overridden=self._selection["raw_student_action"] != action,
                         boundary_hold_interrupted=interrupted,
                         boundary_previous_hold_age=previous_age,
                         boundary_horizon_frames=self.hold_frames,
                         boundary_remaining_check_frames=remaining,
                         boundary_execution_horizon_frames=execution_horizon,
                         boundary_execution_allowed_actions=execution_allowed)
        self.mask_overrides += int(self._selection["boundary_mask_overridden"])
        self.total_overrides += int(self.last["action_overridden"])
        self.interruptions += int(interrupted)
        return action

    def summary(self):
        return {"event": "commitment_summary", "decisions": self.decisions,
                "overridden_decisions": self.total_overrides,
                "mask_overridden_decisions": self.mask_overrides,
                "commitment_overridden_decisions": self.overrides,
                "boundary_hold_interruptions": self.interruptions,
                "hold_frames": self.hold_frames, "boundary_contract": BOUNDARY_CONTRACT}
