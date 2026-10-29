"""Observation-only baselines with the same nine-action interface."""
import math
import random

from env.action_emulator import DIRECTIONS


def make_policy(name, seed=0):
    if name == "random":
        rng = random.Random(seed)
        return lambda obs: rng.randrange(9)
    if name == "stay":
        return lambda obs: 0
    if name == "reactive":
        return reactive
    raise ValueError(f"Unknown policy: {name}")


def reactive(obs):
    """Maximize predicted bullet clearance; fixed low-speed, linear-motion model.

    This baseline ignores lasers, enemies, acceleration, and future spawns.
    It operates on encoded observations, including their clipping/truncation.
    """
    px = (obs["player"][0] + 1) * 192
    py = (obs["player"][1] + 1) * 224
    horizon = 12.0
    best_action, best_score = 0, -math.inf
    for action, keys in enumerate(DIRECTIONS):
        dx = int("right" in keys) - int("left" in keys)
        dy = int("down" in keys) - int("up" in keys)
        scale = 2.0 / max(1.0, math.hypot(dx, dy))
        sx, sy = dx * scale, dy * scale
        end_x, end_y = px + sx * horizon, py + sy * horizon
        if not (8 <= end_x <= 376 and 16 <= end_y <= 432):
            continue
        clearance = 24.0
        for row in obs["bullets"]:
            if not row[6]:
                continue
            bx, by, vx, vy = row[0] * 384, row[1] * 448, row[2] * 16 - sx, row[3] * 16 - sy
            speed2 = vx * vx + vy * vy
            t = max(0.0, min(horizon, -(bx * vx + by * vy) / speed2)) if speed2 else 0.0
            gap = math.hypot(bx + vx * t, by + vy * t) - row[4] * 32 - 1.25
            clearance = min(clearance, gap)
        edge_gap = min(end_x - 8, 376 - end_x, end_y - 16, 432 - end_y)
        score = clearance - max(0.0, 12 - edge_gap) * 0.1 - (0.15 if action else 0)
        if score > best_score:
            best_action, best_score = action, score
    return best_action
