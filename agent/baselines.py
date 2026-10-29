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
    if name == "reactive_attack":
        return reactive_attack
    raise ValueError(f"Unknown policy: {name}")


def reactive(obs):
    """Maximize predicted bullet clearance; fixed low-speed, linear-motion model.

    This baseline ignores lasers, enemies, acceleration, and future spawns.
    It operates on encoded observations, including their clipping/truncation.
    """
    candidates = action_candidates(obs)
    return max(candidates, key=lambda c: c[4])[0] if candidates else 0


def action_candidates(obs):
    """Return action, x/y speed, clearance, and the original reactive score."""
    px = (obs["player"][0] + 1) * 192
    py = (obs["player"][1] + 1) * 224
    horizon = 12.0
    candidates = []
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
        candidates.append((action, sx, sy, clearance, score))
    return candidates


def attack_target(obs):
    """Prefer a visible, collidable boss above the player; ignore padding."""
    targets = [e for e in obs.get("enemies", ()) if e[5] and e[3] and e[4] and e[1] < 0]
    return min(targets, key=lambda e: (-e[2], abs(e[0]), abs(e[1]))) if targets else None


def reactive_attack(obs):
    """Align horizontally only among nearly equally safe predicted paths."""
    target = attack_target(obs)
    if target is None:
        return reactive(obs)
    candidates = action_candidates(obs)
    if not candidates:
        return 0
    best_gap = max(c[3] for c in candidates)
    if best_gap < 4:
        # Tight situations use clearance first, with no attack preference.
        return max(candidates, key=lambda c: (c[3], c[4]))[0]
    safe = [c for c in candidates if c[3] >= max(4, best_gap - 1)]
    tx, ty = target[0] * 384, target[1] * 448
    def attack_score(candidate):
        action, sx, sy, _, score = candidate
        horizontal_error = max(0, abs(tx - sx * 4) - 6)
        too_close = max(0, ty - sy * 4 + 64)
        return (-horizontal_error - too_close * 2, score, -action)
    return max(safe, key=attack_score)[0]
