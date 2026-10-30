"""Action-conditioned geometry features; the actor still chooses every action."""
import math

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env.action_emulator import DIRECTIONS


class HazardFeatures(BaseFeaturesExtractor):
    """Pool bullet trajectories without depending on bullet slot order.

    The 12-frame linear prediction and 2 px/frame speed match the baseline
    approximation. These are geometric inputs, not teacher actions or masks.
    Lasers, acceleration, future spawns, and exact enemy body shapes are absent.
    """

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=62)
        velocities = []
        for keys in DIRECTIONS:
            dx = int("right" in keys) - int("left" in keys)
            dy = int("down" in keys) - int("up" in keys)
            scale = 2 / max(1., math.hypot(dx, dy))
            velocities.append([dx * scale, dy * scale])
        self.register_buffer("velocities", torch.tensor(velocities))

    def forward(self, observations):
        player, bullets, enemies = (observations[k] for k in ("player", "bullets", "enemies"))
        velocity = self.velocities
        px, py = (player[:, 0] + 1) * 192, (player[:, 1] + 1) * 224
        end_x = px[:, None] + velocity[None, :, 0] * 12
        end_y = py[:, None] + velocity[None, :, 1] * 12
        edge = torch.stack((end_x - 8, 376 - end_x, end_y - 16, 432 - end_y), dim=-1).amin(-1)
        valid = edge >= 0

        position = bullets[:, None, :, :2] * bullets.new_tensor([384, 448])
        relative_velocity = bullets[:, None, :, 2:4] * 16 - velocity[None, :, None, :]
        speed2 = relative_velocity.square().sum(-1)
        closest = (-(position * relative_velocity).sum(-1) / speed2.clamp_min(1e-12)).clamp(0, 12)
        closest = torch.where(speed2 > 0, closest, torch.zeros_like(closest))
        gap = (position + relative_velocity * closest[..., None]).norm(dim=-1)
        gap = gap - bullets[:, None, :, 4] * 32 - 1.25
        gap = gap.masked_fill(bullets[:, None, :, 6] <= 0, 24).amin(-1).clamp_max(24)
        best = gap.masked_fill(~valid, -1e6).amax(-1, keepdim=True)
        best = torch.where(valid.any(-1, keepdim=True), best, torch.zeros_like(best))

        eligible = ((enemies[:, :, 5] > 0) & (enemies[:, :, 3] > 0)
                    & (enemies[:, :, 4] > 0) & (enemies[:, :, 1] < 0))
        present = eligible.any(-1)
        boss = enemies[:, :, 2].masked_fill(~eligible, -1).amax(-1, keepdim=True)
        selected = eligible & (enemies[:, :, 2] == boss)
        dx = enemies[:, :, 0].abs().masked_fill(~selected, float("inf")).amin(-1, keepdim=True)
        selected = selected & (enemies[:, :, 0].abs() == dx)
        dy = enemies[:, :, 1].abs().masked_fill(~selected, float("inf")).amin(-1, keepdim=True)
        selected = selected & (enemies[:, :, 1].abs() == dy)
        index = selected.to(torch.int64).argmax(-1)
        target = enemies[torch.arange(len(enemies), device=enemies.device), index, :2]
        target = torch.where(present[:, None], target, torch.zeros_like(target))
        alignment = ((target[:, :1] * 384 - velocity[None, :, 0] * 4).abs() - 6).clamp_min(0)
        approach = (target[:, 1:2] * 448 - velocity[None, :, 1] * 4 + 64).clamp_min(0)
        alignment = torch.where(present[:, None], alignment, torch.zeros_like(alignment))
        approach = torch.where(present[:, None], approach, torch.zeros_like(approach))
        candidates = torch.stack((valid.to(player.dtype), (gap / 24).clamp(-2, 1),
                                  ((gap - best) / 4).clamp(-4, 0), (edge / 16).clamp(-2, 2),
                                  (alignment / 16).clamp(0, 4), (approach / 16).clamp(0, 4)), dim=-1)
        density = (bullets[:, :, 6] > 0).to(player.dtype).mean(-1, keepdim=True)
        return torch.cat((player, target, present[:, None].to(player.dtype), density,
                          candidates.flatten(1)), dim=1)
