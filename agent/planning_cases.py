"""Deterministic fixtures for timed arrival, closure and detour diagnostics."""
from .forward_model import Circle, Scene, TimedLaser
from .timed_planner import TimedGoal


def micro_cases():
    bounds = (0., 40., 0., 40.)
    def wall(on, off, ay=0.):
        return TimedLaser(20., ay, 20., 40., 2., on, off)
    goal = TimedGoal(32., 20., 0, 24, 1.)
    return [
        {"name": "warning_escape", "scene": Scene((8., 20.), lasers=(wall(8, 100),), bounds=bounds),
         "goal": goal, "horizon": 24, "expect_arrival": True},
        {"name": "closed_gate", "scene": Scene((8., 20.), lasers=(wall(0, 100),), bounds=bounds),
         "goal": goal, "horizon": 24, "expect_arrival": False},
        {"name": "wait_for_fade", "scene": Scene((8., 20.), lasers=(wall(0, 12),), bounds=bounds),
         "goal": goal, "horizon": 24, "expect_arrival": True},
        {"name": "detour_before_approach", "scene": Scene((8., 24.), lasers=(wall(0, 100, 12),), bounds=bounds),
         "goal": TimedGoal(32., 24., 0, 32, 1.), "horizon": 32, "expect_arrival": True},
        {"name": "moving_circle", "scene": Scene((8., 20.), circles=(Circle(20., 0., 0., 10., 3.),), bounds=bounds),
         "goal": goal, "horizon": 24, "expect_arrival": True},
        {"name": "arrival_window", "scene": Scene((8., 20.), bounds=bounds),
         "goal": TimedGoal(8., 20., 8, 8, 1.), "horizon": 16, "expect_arrival": True},
    ]
