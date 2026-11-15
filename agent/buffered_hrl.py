"""Offline waypoint PPO with a fixed buffered beam worker."""
from dataclasses import replace
from functools import lru_cache

import gymnasium as gym
import numpy as np

from .direction_buffer import DirectionBuffer
from .forward_model import ForwardModel
from .offline_hrl import EPISODE_FRAMES, OPTION_HOLDS, detour_tasks
from .short_planner import HOLD, VECTORS, plan
from .timed_planner import TimedGoal


WAYPOINT_RADIUS = 2.5
PADDING = 2.
ONE_HOT = np.eye(10, dtype=np.float32)


def buffered_tasks(held_out=False):
    times = (12, 20, 28, 40, 44, 52, 32, 60) if held_out else (16, 24, 32, 48, 56, 32, 24, 60, 36, 20, 48, 56)
    return [replace(task, scene=replace(task.scene, lasers=(replace(task.scene.lasers[0], off=float(off)),)),
                    goal=replace(task.goal, radius=WAYPOINT_RADIUS))
            for task, off in zip(detour_tasks(held_out), times, strict=True)]


def validation_tasks():
    tasks = []
    for index, (task, off) in enumerate(zip(buffered_tasks(True), (14, 22, 30, 38, 46, 54, 26, 58), strict=True)):
        scene, goal = task.scene, task.goal
        dx, dy = -20., 8.
        xmin, xmax, ymin, ymax = scene.bounds
        laser = scene.lasers[0]
        shifted = replace(laser, ax=laser.ax + dx, bx=laser.bx + dx, ay=laser.ay + dy,
                          by=laser.by + dy, off=float(off), width=laser.width + .25)
        scene = replace(scene, player=(scene.player[0] + dx, scene.player[1] + dy), lasers=(shifted,),
                        bounds=(xmin + dx, xmax + dx, ymin + dy, ymax + dy))
        tasks.append(replace(task, name=f'validation-{index}', scene=scene,
                             goal=replace(goal, x=goal.x + dx, y=goal.y + dy)))
    return tasks


class BufferedWorker:
    """Use the frozen goal adapter when its action can still reach the waypoint."""

    def __init__(self, worker):
        self.worker = worker.requires_grad_(False).eval()

    @lru_cache(maxsize=8192)
    def decide(self, scene, goal, buffer, delay):
        padded = replace(scene, player_radius=scene.player_radius + PADDING)
        result = plan(padded, goal, previous_action=buffer.requested, width=9, smooth=False,
                      input_buffer=buffer, switch_delay=delay)
        route = result.selected
        if route is None:
            return None, False
        if route.arrival is not None:
            preferred, _ = self.worker.decide(scene, goal)
            if any(item.arrival is not None and item.actions[0] == preferred for item in result.candidates):
                return preferred, True
        return route.actions[0], False


class BufferedManagerEnv(gym.Env):
    """Choose a waypoint every two holds in a timed laser arena."""

    def __init__(self, tasks, worker, delay=0, verify=False, expose_buffer=True):
        self.tasks, self.worker, self.fixed_delay, self.verify = tasks, worker, delay, verify
        self.expose_buffer = expose_buffer
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0., 1., (38,), dtype=np.float32)

    def _observation(self):
        xmin, _, ymin, _ = self.scene.bounds
        x, y = self.scene.player
        wall_frames = max(0., self.task.scene.lasers[0].off - self.frame)
        pending = self.buffer.pending[0] + 1 if self.buffer.pending else 0
        fields = [(x - xmin) / 40, (y - ymin) / 40, (self.task.goal.x - xmin) / 40,
                  (self.task.goal.y - ymin) / 40, self.task.opening, wall_frames / EPISODE_FRAMES,
                  1 - self.frame / EPISODE_FRAMES, self.delay if self.expose_buffer else 0]
        states = (ONE_HOT[[self.buffer.requested + 1, self.buffer.applied + 1, pending]].ravel()
                  if self.expose_buffer else np.zeros(30, dtype=np.float32))
        return np.concatenate((fields, states)).astype(np.float32).clip(0, 1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = int(options['task']) if options and 'task' in options else int(self.np_random.integers(len(self.tasks)))
        self.task = self.tasks[index]
        self.delay = int(self.np_random.integers(2)) if self.fixed_delay is None else self.fixed_delay
        self.scene, self.frame, self.buffer = self.task.scene, 0, DirectionBuffer()
        self.adapter_choices = 0
        return self._observation(), {}

    def _waypoint(self, option):
        x, y = self.task.goal.x, self.task.goal.y
        if option != 2 and self.scene.lasers and not self.task.crossed(self.scene.player):
            y = self.scene.bounds[2] + (6, 34)[option]
            if abs(self.scene.player[1] - y) > WAYPOINT_RADIUS:
                x = self.scene.player[0]
        return TimedGoal(x, y, 0, 16, WAYPOINT_RADIUS)

    def step(self, action):
        option = int(action)
        old_distance = self.task.remaining(self.scene.player)
        success = blocked = False
        elapsed = 0
        for _ in range(OPTION_HOLDS):
            move, adapter_used = self.worker.decide(self.scene, self._waypoint(option), self.buffer, self.delay)
            if move is None:
                blocked = True
                break
            self.adapter_choices += int(adapter_used)
            directions, self.buffer = self.buffer.advance(move, HOLD, self.delay)
            points = self.scene.player + self.scene.speed * VECTORS[list(directions)].cumsum(axis=0)
            if self.verify:
                reference = ForwardModel(replace(self.scene, player_radius=self.scene.player_radius + PADDING))
                for frame, (start, end) in enumerate(zip(np.vstack((self.scene.player, points)), points)):
                    if not reference.edges([start], [end], frame, 1)[0][0]:
                        raise AssertionError('Worker action failed buffered replay')
            self.frame += HOLD
            elapsed += HOLD
            lasers = tuple(replace(laser, on=max(0., laser.on - HOLD), off=laser.off - HOLD)
                           for laser in self.scene.lasers if laser.off > HOLD)
            self.scene = replace(self.scene, player=tuple(map(float, points[-1])), lasers=lasers)
            success = self.task.goal.reached(self.scene.player, self.frame)
            if success or self.frame >= EPISODE_FRAMES:
                break
        done = success or blocked or self.frame >= EPISODE_FRAMES
        progress = np.clip((old_distance - self.task.remaining(self.scene.player)) / 40, -.5, .5)
        reward = -.015 * elapsed / HOLD + .3 * float(progress)
        reward += 2. if success else -.5 if done else 0.
        return self._observation(), reward, done, False, {'success': success, 'blocked': blocked,
                                                          'frame': self.frame, 'adapter_choices': self.adapter_choices}
