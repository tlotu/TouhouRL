"""Closed-loop waypoint selection over recorded Young Demon Lord hazards."""
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from functools import lru_cache
import time

import gymnasium as gym
import numpy as np

from .boundary_control import feasible_actions
from .forward_model import ForwardModel
from .short_planner import ANGLES, HOLD, SHORT_HORIZON, TTC_HORIZON, VECTORS, Route, SearchBudget, plan
from .swept_collision import SweptCollision


PADDING = 2.
WIDTH = 9
MAX_EDGES = 900
SCALE = np.array([192., 224.])
EPISODE_WINDOW = 96
OPTIONS = len(VECTORS)
FEATURES = 4 + 4 + 3 * OPTIONS + OPTIONS * 7
RISK_THRESHOLD = 20
RISK_DELAY = 32
SURVIVAL_HORIZON = 60
ROUTE_DROP = .20
NARROW_EXIT_COUNT = 3
ATTACK_BAND = 24.
EARLY_EXIT_GAP = 8.
EARLY_EXIT_DIRECTIONS = 3


def boss_offset(observation):
    """Return the first targetable boss's relative position in pixels."""
    enemies = observation['enemies']
    bosses = enemies[(enemies[:, 2] > 0) & (enemies[:, 3:6] > 0).all(axis=1)]
    return bosses[0, :2] * (2 * SCALE) if len(bosses) else None


def escape_capacity(choices):
    """Count distinct verified first moves with at least 20 frames of leaf TTC."""
    return (len({choice.route.actions[0] for choice in choices if choice.ttc >= RISK_THRESHOLD}),
            max((choice.ttc for choice in choices), default=0.))


def routes_contract(previous, current, warning):
    count, ttc = current
    closing = warning <= SURVIVAL_HORIZON - SHORT_HORIZON and count <= NARROW_EXIT_COUNT
    if previous is None:
        return closing or ttc < RISK_THRESHOLD
    old_count, old_ttc = previous
    return (closing or ttc < RISK_THRESHOLD
            or (count < old_count and count <= (1 - ROUTE_DROP) * old_count)
            or (ttc < old_ttc and ttc <= (1 - ROUTE_DROP) * old_ttc))


@dataclass
class ReplaySample:
    observation: dict
    scene: object
    omitted: int


@dataclass
class ReplayEpisode:
    source: str
    number: int
    samples: dict
    priors: dict
    end: int

    def start_frames(self):
        return [frame for frame in sorted(self.priors)
                if frame % HOLD == 0 and frame + EPISODE_WINDOW <= self.end]


def rebase(obs, position):
    """Keep recorded absolute geometry while updating player-relative actor rows."""
    displacement = np.asarray(position) - (obs['player'][:2].astype(float) + 1.) * SCALE
    result = dict(obs)
    player = obs['player'].copy()
    player[:2] = np.asarray(position) / SCALE - 1.
    result['player'] = player
    for key in ('bullets', 'enemies'):
        rows = obs[key].copy()
        present = rows[:, -1] > 0
        rows[present, :2] = np.clip(rows[present, :2] - displacement / (2 * SCALE), -1., 1.)
        result[key] = rows
    return result


@lru_cache(maxsize=8192)
def route_motions(actions, belief):
    motions = {((), state) for state in belief}
    for action in actions:
        motions = {(prefix + moves, tail) for prefix, state in motions for delay in (0, 1)
                   for moves, tail in (state.advance(action, HOLD, delay),)}
    return tuple(sorted({moves for moves, _ in motions}))


def hold_outcomes(scene, belief, action):
    """Map observable endpoints to the queue states consistent with each."""
    endpoints = defaultdict(set)
    for state in belief:
        for delay in (0, 1):
            moves, tail = state.advance(action, HOLD, delay)
            point = scene.player + scene.speed * VECTORS[list(moves)].sum(axis=0)
            endpoints[tuple(np.round(point, 6))].add(tail)
    return endpoints


def robust_routes(scene, belief, routes, allowed, model, *, prune_failed=False):
    """Verify all delay sequences for each candidate with the padded model."""
    pairs = [(i, moves) for i, route in enumerate(routes) if route.actions[0] in allowed
             for moves in route_motions(route.actions, belief)]
    if not pairs:
        return []
    # Many action sequences produce identical movement under buffered input.
    indices, motions = zip(*pairs)
    indices = np.asarray(indices)
    unique = {motion: i for i, motion in enumerate(dict.fromkeys(motions))}
    mapping = np.asarray([unique[motion] for motion in motions])
    points = scene.player + scene.speed * VECTORS[np.asarray(tuple(unique))].cumsum(axis=1)
    starts = np.broadcast_to(scene.player, (len(points), 2))
    safe = np.ones(len(points), dtype=bool)
    clearance = np.full(len(points), model.clearance_cap)
    good = np.ones(len(routes), dtype=bool)
    active = slice(None)
    for frame, ends in enumerate(points.transpose(1, 0, 2)):
        legal, gap = model.edges(starts[active], ends[active], frame, 1)
        safe[active] &= legal
        clearance[active] = np.minimum(clearance[active], gap)
        if prune_failed and not legal.all():
            # A route needs every delay branch. Retain shared motions only
            # while at least one surviving route still needs them.
            np.logical_and.at(good, indices, safe[mapping])
            needed = np.zeros(len(points), dtype=bool)
            needed[mapping[good[indices]]] = True
            active = np.flatnonzero(needed)
            if not len(active):
                return []
        starts = ends
    times = np.full(len(points), TTC_HORIZON, dtype=float)
    times[active] = model.stationary_ttc(points[active, -1], SHORT_HORIZON, TTC_HORIZON)
    worst_gap = np.full(len(routes), model.clearance_cap)
    worst_ttc = np.full(len(routes), TTC_HORIZON, dtype=float)
    np.logical_and.at(good, indices, safe[mapping])
    np.minimum.at(worst_gap, indices, clearance[mapping])
    np.minimum.at(worst_ttc, indices, times[mapping])
    return [(routes[i], worst_gap[i], worst_ttc[i]) for i in sorted(set(indices)) if good[i]]


@dataclass
class Choice:
    route: object
    gap: float
    ttc: float
    target: np.ndarray


class ReplayWorker:
    """Frozen actor and deterministic short search shared by all managers."""

    def __init__(self, logits, *, early_exit=False, prune_failed=False):
        self.logits = logits
        self.early_exit = early_exit
        self.prune_failed = prune_failed

    def _direct_routes(self, scene, state, allowed, belief, collision):
        """Try sustained directions before expanding a beam in open space."""
        candidates = []
        for action in allowed:
            moves, _ = state.advance(action, SHORT_HORIZON)
            point = np.asarray(scene.player) + scene.speed * VECTORS[list(moves)].sum(axis=0)
            candidates.append(Route((), tuple(point), (action,) * (SHORT_HORIZON // HOLD),
                                    turn=ANGLES[state.requested, action]))
        verified = robust_routes(scene, belief, candidates, allowed, collision)
        roomy = [entry for entry in verified if entry[1] >= EARLY_EXIT_GAP and entry[2] >= TTC_HORIZON]
        return roomy if len(roomy) >= EARLY_EXIT_DIRECTIONS else []

    def options(self, scene, observation, belief, *, by_direction=False, waypoint_coverage=False):
        started = time.perf_counter()
        padded = replace(scene, player_radius=scene.player_radius + PADDING)
        collision = SweptCollision(padded)
        proposals = {}
        # One representative queue generates candidates. The full belief is used
        # for verification, so additional queue states can only remove candidates.
        state = min(belief, key=lambda q: (q.requested, q.applied, q.pending))
        allowed = feasible_actions(observation, HOLD)
        if self.early_exit and not (by_direction or waypoint_coverage) and not any(
                0 < laser.on <= SURVIVAL_HORIZON for laser in padded.lasers):
            direct = self._direct_routes(padded, state, allowed, belief, collision)
            if direct:
                return self._rank(scene, observation, direct), 0, 1000 * (time.perf_counter() - started)
        edges = 0
        for delay in (0, 1):
            result = plan(padded, previous_action=state.requested, width=WIDTH, smooth=False,
                          budget=SearchBudget(MAX_EDGES), input_buffer=state, switch_delay=delay,
                          ttc_weight=0., collision_model=collision)
            edges += result.checked_edges
            for route in result.candidates:
                if len(route.actions) * HOLD == SHORT_HORIZON:
                    proposals.setdefault(route.actions, route)
        routes = robust_routes(padded, belief, tuple(proposals.values()), allowed, collision,
                               prune_failed=self.prune_failed)
        baseline = self._rank(scene, observation, routes) if waypoint_coverage and routes else ()
        if by_direction or waypoint_coverage:
            covered = {route.actions[0] for route, _, _ in routes}
            extra = {}
            for action in sorted(set(allowed) - covered):
                for delay in (0, 1):
                    result = plan(padded, previous_action=state.requested, width=WIDTH, smooth=False,
                                  input_buffer=state, switch_delay=delay, ttc_weight=0.,
                                  collision_model=collision, first_action=action)
                    edges += result.checked_edges
                    for route in result.candidates:
                        if len(route.actions) * HOLD == SHORT_HORIZON:
                            extra.setdefault(route.actions, route)
            routes.extend(robust_routes(padded, belief, tuple(extra.values()), allowed, collision,
                                        prune_failed=True))
            directions = self._rank(scene, observation, routes, by_direction=True)
            if by_direction:
                return directions, edges, 1000 * (time.perf_counter() - started)
            choices = self._assign_waypoints(scene, baseline, directions)
            return choices, edges, 1000 * (time.perf_counter() - started)
        if not routes:
            return (), edges, 1000 * (time.perf_counter() - started)
        return self._rank(scene, observation, routes), edges, 1000 * (time.perf_counter() - started)

    def _rank(self, scene, observation, routes, *, by_direction=False):
        prior = self.logits(observation)
        prior = np.clip(np.asarray(prior), -6., 6.)

        def safety(entry):
            route, gap, ttc = entry
            return float(gap + .25 * ttc - .5 * route.turn + .1 * prior[route.actions[0]])

        def pick(target, directional):
            # Goals bias only the ranking of routes already verified for 16 frames.
            route, gap, ttc = max(routes, key=lambda r: safety(r) -
                                  (.45 * np.linalg.norm(r[0].point - target) if directional else 0.))
            return Choice(route, float(gap), float(ttc), target if directional else np.asarray(route.point))

        if by_direction:
            best = {}
            for entry in routes:
                direction = entry[0].actions[0]
                if direction not in best or safety(entry) > safety(best[direction]):
                    best[direction] = entry
            return {direction: Choice(route, float(gap), float(ttc), np.asarray(route.point))
                    for direction, (route, gap, ttc) in best.items()}
        return (pick(np.asarray(scene.player), False),) + tuple(
            pick(scene.player + vector * scene.speed * SHORT_HORIZON, True) for vector in VECTORS[1:])

    def _assign_waypoints(self, scene, baseline, directions):
        if not directions:
            return ()
        rule = baseline[0] if baseline else max(directions.values(), key=lambda c: (c.ttc, c.gap))
        choices = list(baseline) if baseline else [rule] * OPTIONS
        remaining = set(range(1, OPTIONS))
        moves = set(directions) - {rule.route.actions[0]}
        while moves:
            slot, move = min(((slot, move) for slot in remaining for move in moves),
                             key=lambda pair: (np.linalg.norm(
                                 VECTORS[pair[0]] * SHORT_HORIZON * scene.speed
                                 - (directions[pair[1]].target - scene.player)), *pair))
            target = scene.player + VECTORS[slot] * SHORT_HORIZON * scene.speed
            choices[slot] = replace(directions[move], target=target)
            remaining.remove(slot)
            moves.remove(move)
        return tuple(choices)


class ReplayManagerEnv(gym.Env):
    """Select a waypoint, execute its first hold, and observe the new position."""

    def __init__(self, episodes, worker, verify=False, long_reward=False, *,
                 conditional_reward=False, attack_reward=False):
        self.episodes, self.worker, self.verify = episodes, worker, verify
        self.long_reward = long_reward
        self.conditional_reward = conditional_reward
        self.attack_reward = attack_reward or conditional_reward
        self.starts = [(i, frame) for i, ep in enumerate(episodes) for frame in ep.start_frames()]
        self.action_space = gym.spaces.Discrete(OPTIONS)
        self.observation_space = gym.spaces.Box(-1., 1., (FEATURES,), dtype=np.float32)

    def _prepare(self):
        sample = self.episode.samples[self.frame]
        self.scene = replace(sample.scene, player=tuple(self.position))
        obs = rebase(sample.observation, self.position)
        self.choices, edges, ms = self._plan_options(obs)
        capacity = escape_capacity(self.choices)
        warning = min((laser.on for laser in self.scene.lasers if laser.on > 0), default=float('inf'))
        self.danger = routes_contract(self.previous_capacity, capacity, warning)
        self.previous_capacity = capacity
        self.boss = boss_offset(obs) if self.attack_reward else None
        self.times.append(ms)
        self.edges.append(edges)
        self.max_omitted = max(self.max_omitted, sample.omitted)
        return self._observation(obs, sample)

    def _plan_options(self, obs):
        return self.worker.options(self.scene, obs, self.belief)

    def _observation(self, obs, sample):
        scene = self.scene
        warnings = [laser.on for laser in scene.lasers if laser.on > 0]
        fields = [min(len(warnings), 64) / 64, min(warnings, default=60.) / 60,
                  len(scene.circles) / 128, min(sample.omitted, 128) / 128]
        queues = np.zeros((3, OPTIONS))
        for state in self.belief:
            queues[0, state.requested] += 1.
            queues[1, state.applied] += 1.
            if state.pending:
                queues[2, state.pending[0]] += 1.
        queue_features = queues.ravel() / len(self.belief)
        choices = np.zeros((OPTIONS, 7))
        for i, choice in enumerate(self.choices):
            if choice is None:
                continue
            route = choice.route
            choices[i] = [*(choice.target - self.position) / 32.,
                          *(np.asarray(route.point) - self.position) / 32.,
                          choice.gap / 24, choice.ttc / TTC_HORIZON, route.actions[0] / 8]
        return np.concatenate((obs['player'], fields, queue_features, choices.ravel())).clip(-1, 1).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is None:
            index, frame = self.starts[int(self.np_random.integers(len(self.starts)))]
            options = {'episode': index, 'frame': frame, 'frames': EPISODE_WINDOW}
        self.episode = self.episodes[options['episode']]
        self.frame = self.start = options.get('frame', 0)
        self.limit = min(self.episode.end, self.start + options.get('frames', self.episode.end))
        self.fixed_delay = options.get('delay')
        self.belief = self.episode.priors[self.frame]
        ordered = sorted(self.belief, key=lambda q: (q.requested, q.applied, q.pending))
        self.buffer = ordered[int(self.np_random.integers(len(ordered)))]
        self.position = np.asarray(self.episode.samples[self.frame].scene.player)
        self.times, self.edges, self.history = [], [], []
        self.max_omitted = 0
        self.min_gap = float('inf')
        self.padded_contact = None
        self.last_padded_contact = None
        self.forecasts = deque()
        self.delayed_checks = deque()
        self.distinct_moves = 0
        self.previous_capacity = None
        return self._prepare(), {}

    def _long_reward(self, outcome):
        """Score continuation at +32 and confirm forecasts against +60 of replay."""
        penalty = bonus = 0.
        if self.delayed_checks and self.frame >= self.delayed_checks[0] + RISK_DELAY:
            self.delayed_checks.popleft()
            if outcome not in ('trace_end', 'window_end'):
                best = max((choice.ttc for choice in self.choices), default=0.) if outcome is None else 0.
                scale = .12 if self.conditional_reward else .06
                penalty = -scale * max(0., 1. - best / RISK_THRESHOLD)
        while self.forecasts and self.frame >= self.forecasts[0] + SURVIVAL_HORIZON:
            origin = self.forecasts.popleft()
            if self.last_padded_contact is None or self.last_padded_contact <= origin:
                bonus += .04
        return penalty, bonus

    def step(self, action):
        outcome = None
        executed = 0
        danger = self.danger and bool(self.choices)
        attack = 0.
        start_frame = self.frame
        boss = self.position + self.boss if self.boss is not None else None
        if not self.choices:
            outcome = 'no_verified_route'
        else:
            choice = self.choices[int(action)]
            move = choice.route.actions[0]
            if self.long_reward or (self.conditional_reward and danger):
                self.delayed_checks.append(self.frame)
                if SHORT_HORIZON + choice.ttc >= SURVIVAL_HORIZON:
                    self.forecasts.append(self.frame)
            self.distinct_moves += int(len({c.route.actions[0] for c in self.choices}) > 1)
            delay = self.fixed_delay if self.fixed_delay is not None else int(self.np_random.integers(2))
            moves, self.buffer = self.buffer.advance(move, HOLD, delay)
            outcomes = hold_outcomes(self.scene, self.belief, move)
            if self.verify:
                padded = replace(self.scene, player_radius=self.scene.player_radius + PADDING)
                reference = ForwardModel(padded)
                motions = np.asarray(route_motions(choice.route.actions, self.belief))
                points = self.position + self.scene.speed * VECTORS[motions].cumsum(axis=1)
                starts = np.broadcast_to(self.position, (len(points), 2))
                for offset, ends in enumerate(points.transpose(1, 0, 2)):
                    if not reference.edges(starts, ends, offset, 1)[0].all():
                        raise AssertionError('Route failed independent verification')
                    starts = ends
            start = self.position.copy()
            for direction in moves[:self.limit - self.frame]:
                # Samples are refreshed at the recorded cadence, even inside a hold.
                sample_frame = self.frame if self.frame in self.episode.samples else self.frame - 1
                sample = self.episode.samples[sample_frame]
                self.max_omitted = max(self.max_omitted, sample.omitted)
                end = self.position + self.scene.speed * VECTORS[direction]
                safe, gap = ForwardModel(sample.scene).edges([self.position], [end], self.frame - sample_frame, 1)
                self.position = end
                self.frame += 1
                executed += 1
                self.min_gap = min(self.min_gap, float(gap[0]))
                if gap[0] <= PADDING:
                    self.last_padded_contact = self.frame
                    if self.padded_contact is None:
                        self.padded_contact = self.frame
                if not safe[0]:
                    outcome = 'collision'
                    break
            if self.verify:
                self.history.append(dict(frame=self.frame - executed, option=int(action), request=move,
                                         start=start.tolist(), end=self.position.tolist(),
                                         waypoint=choice.target.tolist(), leaf_ttc=choice.ttc,
                                         danger=danger, escape_directions=self.previous_capacity[0]))
            if executed == HOLD:
                self.belief = frozenset(outcomes[tuple(np.round(self.position, 6))])
                if self.buffer not in self.belief:
                    raise AssertionError('Observed endpoint removed the simulated input queue')
        if outcome is None and self.frame == self.limit:
            outcome = 'trace_end' if self.frame == self.episode.end else 'window_end'
        if self.attack_reward and boss is not None and not (self.conditional_reward and danger):
            if self.position[1] >= boss[1]:
                excess = max(0., abs(self.position[0] - boss[0]) - ATTACK_BAND)
                attack = .015 * executed / HOLD * max(0., 1. - excess / 64.)
        obs = self._prepare() if outcome is None else np.zeros(FEATURES, dtype=np.float32)
        reward = .01 * executed - (1. if outcome in ('collision', 'no_verified_route') else 0.)
        penalty, bonus = self._long_reward(outcome) if self.long_reward or self.conditional_reward else (0., 0.)
        if self.conditional_reward and outcome in ('collision', 'no_verified_route'):
            # Charge a dead end once if a recently flagged decision has not matured.
            if self.delayed_checks and start_frame - self.delayed_checks[-1] <= RISK_DELAY:
                penalty = min(penalty, -.12)
        reward += penalty + bonus + attack
        info = dict(outcome=outcome, frames=self.frame - self.start, start=self.start,
                    continuation_penalty=penalty, survival_bonus=bonus,
                    danger=danger, attack_reward=attack,
                    distinct_move_decisions=self.distinct_moves,
                    min_clearance_px=None if not np.isfinite(self.min_gap) else self.min_gap,
                    first_below_padding_frame=self.padded_contact, max_omitted_circles=self.max_omitted,
                    planning_ms=self.times, checked_edges=self.edges, history=self.history)
        return obs, reward, outcome is not None, False, info
