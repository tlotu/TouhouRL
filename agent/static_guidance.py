"""Timed subgoals from a checked route through persistent obstacles."""
from dataclasses import replace

from .forward_model import ForwardModel
from .short_planner import HOLD
from .timed_planner import TimedGoal, path_points, replay_path, search


def static_waypoints(scene, goal):
    """Return turn points and short subgoals for a static, reachable goal."""
    if goal is None or scene.circles or not scene.lasers:
        return ()
    if any(laser.on != 0 or laser.off <= goal.deadline for laser in scene.lasers):
        return ()
    horizon = goal.deadline // HOLD * HOLD
    if not HOLD <= horizon <= 256:
        return ()
    # Avoid searching for a detour when the direct line is already clear.
    destination = (goal.x, goal.y)
    if ForwardModel(scene).edges([scene.player], [destination], 0, horizon)[0][0]:
        return ()
    result = search(scene, horizon, goal=goal)
    route = result.goal_path
    if route is None:
        return ()
    if not replay_path(scene, route):
        raise AssertionError('Reference guide failed collision replay')
    points = path_points(scene, route)
    waypoints = []
    last = 0
    for index in range(1, len(route.actions)):
        turns = route.actions[index] != route.actions[index - 1]
        if turns or index - last == 2:
            waypoints.append(TimedGoal(*points[index], 0, index * HOLD, min(1., goal.radius)))
            last = index
    return (*waypoints, goal)


def relative_goal(goal, frame):
    if goal is None or frame > goal.deadline:
        return None
    return replace(goal, earliest=max(0, goal.earliest - frame), deadline=goal.deadline - frame)
