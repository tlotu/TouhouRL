"""Small direction scorer for the two-proposal Young Demon Lord replay search."""
import numpy as np
import torch
from torch import nn

from .forward_model import _point_segment
from .replay_hrl import PADDING, boss_offset
from .short_planner import VECTORS


def direction_features(scene, observation, belief, prior):
    direction = VECTORS
    prior = np.asarray(prior)
    boss = boss_offset(observation)
    fields = [direction.T[0], direction.T[1], (prior - prior.max()).clip(-12, 0) / 6]
    for attr in ('applied', 'requested'):
        counts = np.bincount([getattr(q, attr) for q in belief], minlength=9) / len(belief)
        fields.extend([counts, direction @ (counts @ direction)])
    pending = np.bincount([q.pending[0] for q in belief if q.pending], minlength=9) / len(belief)
    fields.append(pending)
    circles = np.asarray([(c.x, c.y, c.vx, c.vy, c.radius) for c in scene.circles]).reshape(-1, 5)
    beams = np.asarray([(l.ax, l.ay, l.bx, l.by, l.width / 2, l.on, l.off)
                        for l in scene.lasers if l.off > l.on]).reshape(-1, 7)
    radius = scene.player_radius + PADDING
    xmin, xmax, ymin, ymax = scene.bounds
    for frame in (4, 8, 16, 32):
        point = scene.player + frame * scene.speed * direction
        x, y = point.T
        fields.append(np.minimum.reduce((x - xmin, xmax - x, y - ymin, ymax - y)) / 48)
        fields.append(np.abs(boss[0] - frame * scene.speed * direction[:, 0]) / 96
                      if boss is not None else np.ones(9))
        # Cheap geometric samples, never used as a collision certificate.
        if len(circles):
            centers = circles[:, :2] + frame * circles[:, 2:4]
            gaps = np.linalg.norm(point[:, None] - centers, axis=2) - circles[:, 4] - radius
            fields.append(gaps.min(axis=1) / 32)
        else:
            fields.append(np.full(9, 2.))
        if len(beams):
            gaps = _point_segment(point[:, None], beams[:, :2], beams[:, 2:4]) - beams[:, 4] - radius
            active = (beams[:, 5] <= frame) & (beams[:, 6] > frame)
            warning = (frame < beams[:, 5]) & (beams[:, 5] <= frame + 24)
            for mask in (active, warning):
                fields.append(gaps[:, mask].min(axis=1) / 32 if mask.any() else np.full(9, 2.))
        else:
            fields.extend([np.full(9, 2.)] * 2)
    return np.stack(fields, axis=1).clip(-2., 2.).astype(np.float32)


class DirectionRanker(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(64, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, x):
        action = self.encoder(x)
        context = action.amax(dim=-2, keepdim=True).expand_as(action)
        return self.readout(torch.cat((action, context), dim=-1)).squeeze(-1)
