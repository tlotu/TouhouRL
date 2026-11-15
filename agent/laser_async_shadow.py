"""Native shadow adapter for the four-frame speculative planner."""
import numpy as np

from .direction_buffer import DirectionBuffer
from .forward_model import scene_from_observation
from .laser_shadow import LaserBufferedReuseCommitment
from .replay_async import AsyncReplayPlanner, fallback_hold
from .replay_hrl import HOLD, ReplayWorker


ASYNC_SHADOW_CONTRACT = 'laser-async-shadow-v1'


class LaserAsyncShadowCommitment(LaserBufferedReuseCommitment):
    """Keep baseline input and log completed, revalidated proposals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = AsyncReplayPlanner(ReplayWorker(lambda _: np.zeros(9)))
        self.belief = frozenset({DirectionBuffer()})

    def close(self):
        self.planner.close()

    def annotate(self, row):
        if row['event'] == 'reset':
            self.planner.reset()
            self.belief = frozenset({DirectionBuffer()})
        if row['event'] == 'transition':
            self.belief = frozenset(
                state.advance(row['action'], row['info']['delta_frames'], delay)[1]
                for state in self.belief for delay in (0, 1))
        return super().annotate(row)

    def __call__(self, obs):
        frame = self.elapsed_frames()
        action = self._baseline_action(obs, frame)
        record = {'status': 'held', 'submitted': False, 'suggested_action': None}
        started = self.clock()
        if not self.last['commitment_active']:
            scene, _ = scene_from_observation(obs)
            choice, status = self.planner.consume(frame, scene, obs, self.belief)
            source = 'async'
            if choice is None:
                choice = fallback_hold(scene, obs, self.belief, self.buffer.requested)
                source = 'hold_gate' if choice else 'unavailable'
            record.update(status=status, source=source, suggested_action=choice.route.actions[0] if choice else None,
                          submitted=self.planner.submit(frame, scene, obs, self.belief, action),
                          target_frame=frame + HOLD)
        record['main_thread_seconds'] = self.clock() - started
        self.last['planner_async_shadow'] = record
        return action
