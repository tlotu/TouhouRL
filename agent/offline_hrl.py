"""Goal-conditioned worker and fixed-duration manager for static model arenas."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .forward_model import ForwardModel, Scene, TimedLaser
from .replay_data import decision_record
from .short_planner import HOLD, VECTORS
from .swept_collision import SweptCollision
from .timed_planner import TimedGoal


OPTION_HOLDS = 2
EPISODE_FRAMES = 64
GOAL_RADIUS = 1.25


@dataclass(frozen=True)
class DetourTask:
    name: str
    scene: Scene
    goal: TimedGoal
    opening: int

    def crossed(self, point):
        wall = self.scene.lasers[0].ax
        return (point[0] - wall) * (self.goal.x - wall) > 0

    def option(self, scene, choice):
        if choice == 2:
            x, y = self.goal.x, self.goal.y
        else:
            y = scene.bounds[2] + (8, 32)[choice]
            x = self.goal.x if abs(scene.player[1] - y) <= GOAL_RADIUS else scene.player[0]
        return TimedGoal(x, y, 0, OPTION_HOLDS * HOLD, GOAL_RADIUS)

    def remaining(self, point):
        x, y = point
        if self.crossed(point):
            return abs(x - self.goal.x) + abs(y - self.goal.y)
        lane = self.scene.bounds[2] + (8, 32)[self.opening]
        return abs(x - self.goal.x) + abs(y - lane) + abs(lane - self.goal.y)


def detour_tasks(held_out=False):
    layouts = ((112, 308, 24, 3.), (144, 328, 16, 1.)) if held_out else (
        (96, 288, 16, 1.5), (128, 320, 24, 2.), (160, 336, 16, 2.5))
    tasks = []
    for index, (x, y, row, width) in enumerate(layouts):
        for opening, (top, bottom) in enumerate(((12, 40), (0, 28))):
            wall = TimedLaser(x + 20, y + top, x + 20, y + bottom, width, 0, 96)
            for side, (start, end) in enumerate(((8, 32), (32, 8))):
                scene = Scene((x + start, y + row), lasers=(wall,), bounds=(x, x + 40, y, y + 40))
                goal = TimedGoal(x + end, y + row, 0, EPISODE_FRAMES, GOAL_RADIUS)
                tasks.append(DetourTask(f'{index}-opening{opening}-side{side}', scene, goal, opening))
    return tasks


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GoalWorker(nn.Module):
    """A learned goal adapter on top of frozen checkpoint logits."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapter = nn.Sequential(nn.Linear(39, 64), nn.Tanh(), nn.Linear(64, 9))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    @lru_cache(maxsize=1024)
    @torch.no_grad()
    def geometry(self, scene):
        obs = {key: np.zeros(space.shape, dtype=np.float32)
               for key, space in self.base.observation_space.spaces.items()}
        x, y = scene.player
        obs['player'][:2] = (x / 192 - 1, y / 224 - 1)
        for row, beam in zip(obs['lasers'], scene.lasers):
            row[:] = [beam.ax, beam.ay, beam.bx, beam.by, beam.width, 1, 0, 0, beam.off, 0, 0, 1]
        tensor, _ = self.base.obs_to_tensor(obs)
        prior = self.base.unmasked_logits(tensor)[0].detach().numpy()
        starts = np.broadcast_to(scene.player, (9, 2))
        endpoints = starts + VECTORS * scene.speed * HOLD
        safe, gaps = SweptCollision(scene).edges(starts, endpoints, 0, HOLD)
        return prior, endpoints, safe, gaps

    def example(self, scene, goal):
        prior, endpoints, safe, gaps = self.geometry(scene)
        delta = np.array((goal.x, goal.y)) - scene.player
        distance = np.linalg.norm(endpoints - (goal.x, goal.y), axis=1)
        progress = (np.linalg.norm(delta) - distance) / (scene.speed * HOLD)
        features = np.concatenate((np.clip(prior, -6, 6), gaps.clip(-24, 24) / 24, safe, progress,
                                   np.clip(delta / 24, -2, 2), [goal.radius / 8])).astype(np.float32)
        return features, prior, safe, distance

    def logits(self, features, prior, safe):
        return (prior + self.adapter(features)).masked_fill(~safe, -torch.inf)

    @torch.no_grad()
    def decide(self, scene, goal):
        features, prior, safe, _ = self.example(scene, goal)
        if not safe.any():
            return None, None
        logits = self.logits(torch.from_numpy(features), torch.from_numpy(prior), torch.from_numpy(safe))
        return int(logits.argmax()), logits.softmax(0).tolist()


def worker_examples(worker, tasks, count, seed):
    rng = np.random.default_rng(seed)
    examples, labels = [], []
    for _ in range(count):
        task = tasks[int(rng.integers(len(tasks)))]
        xmin, _, ymin, _ = task.scene.bounds
        player = (xmin + float(rng.choice([0, 8, 16, 24, 32, 40])),
                  ymin + float(rng.choice([0, 8, 16, 24, 32, 40])))
        scene = replace(task.scene, player=player)
        goal = task.option(scene, int(rng.integers(3)))
        if rng.random() < .5:
            goal = TimedGoal(xmin + float(rng.uniform(0, 40)), ymin + float(rng.uniform(0, 40)),
                             0, HOLD, GOAL_RADIUS)
        features, prior, safe, distance = worker.example(scene, goal)
        if not safe.any():
            continue
        action = int(np.argmin(np.where(safe, distance, np.inf)))
        if safe[0] and distance[0] <= goal.radius:
            action = 0
        examples.append((features, prior, safe))
        labels.append(action)
    columns = tuple(torch.from_numpy(np.stack(column)) for column in zip(*examples))
    return (*columns, torch.tensor(labels))


class DetourManagerEnv(gym.Env):
    """The manager chooses a lane or the destination every two held actions."""

    def __init__(self, tasks, worker, record=False):
        self.tasks, self.worker, self.record = tasks, worker, record
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0., 1., (6,), dtype=np.float32)

    def _observation(self):
        xmin, _, ymin, _ = self.scene.bounds
        x, y = self.scene.player
        return np.array([(x - xmin) / 40, (y - ymin) / 40, (self.task.goal.x - xmin) / 40,
                         (self.task.goal.y - ymin) / 40, self.task.opening, 1 - self.frame / EPISODE_FRAMES],
                        dtype=np.float32).clip(0, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(options['task']) if options and 'task' in options else int(self.np_random.integers(len(self.tasks)))
        self.task = self.tasks[index]
        self.scene, self.frame, self.decisions = self.task.scene, 0, []
        return self._observation(), {}

    def step(self, action):
        goal = self.task.option(self.scene, int(action))
        remaining = self.task.remaining(self.scene.player)
        reward, success, blocked = 0., False, False
        for _ in range(OPTION_HOLDS):
            move, probabilities = self.worker.decide(self.scene, goal)
            if self.record:
                decision = decision_record(self.scene, self.frame, goal, action=move, probabilities=probabilities)
                decision['manager_action'] = int(action)
                self.decisions.append(decision)
            if move is None:
                blocked = True
                break
            end = tuple(map(float, np.asarray(self.scene.player) + VECTORS[move] * self.scene.speed * HOLD))
            if self.record and not ForwardModel(self.scene).edges([self.scene.player], [end], 0, HOLD)[0][0]:
                raise AssertionError('Worker action failed independent collision replay')
            self.scene = replace(self.scene, player=end)
            self.frame += HOLD
            reward -= .015
            success = self.task.goal.reached(end, self.frame)
            if success or self.frame >= EPISODE_FRAMES:
                break
        done = success or blocked or self.frame >= EPISODE_FRAMES
        progress = (remaining - self.task.remaining(self.scene.player)) / 40
        reward += .3 * float(np.clip(progress, -.5, .5))
        reward += 2. if success else -.5 if done else 0.
        return self._observation(), reward, done, False, {'success': success, 'blocked': blocked, 'frame': self.frame}
