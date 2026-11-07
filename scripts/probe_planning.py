"""Offline planning diagnostics. This module never opens the game or sends keys."""
import argparse
from collections import deque
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
import traceback

import numpy as np

from agent.forward_model import MODEL_CONTRACT, ForwardModel, scene_from_observation
from agent.planning_cases import micro_cases
from agent.timed_planner import PLANNER_CONTRACT, path_points, replay_path, search


LIMITATIONS = [
    "Offline open-loop hypotheses; no native input, training or counterfactual game outcome.",
    "Only current encoded circles and static laser axes; no future births or acceleration.",
    "Five held cardinal/stay actions, not all nine native actions; reference is complete only on this lattice.",
    "Beam pruning can miss feasible routes. Unvisited positions are unknown, not necessarily unsafe.",
    "Room areas use laser-only grid components at earliest peak activation; no survival or damage probability.",
    "Goal arrival does not imply survival after arrival. A full-horizon path proves only model feasibility.",
    "Capsules, radius, speed and phase timing remain approximate native collision hypotheses.",
    "Policy observation clipping/omission cannot be reconstructed. No source-forecast calibration yet.",
]


def laser_regions(scene, frame, spacing=4.):
    """One laser-only future partition; never used as a hard collision filter."""
    xmin, xmax, ymin, ymax = scene.bounds
    xs, ys = np.arange(xmin + spacing / 2, xmax, spacing), np.arange(ymin + spacing / 2, ymax, spacing)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    free, _ = ForwardModel(replace(scene, circles=())).edges(points, points, frame, 0)
    labels = np.zeros(xx.shape, dtype=int)
    free = free.reshape(xx.shape)
    areas = [0.]
    for y, x in np.argwhere(free):
        if labels[y, x]:
            continue
        label, count = len(areas), 0
        labels[y, x] = label
        queue = deque([(y, x)])
        while queue:
            iy, ix = queue.popleft()
            count += 1
            for ny, nx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
                if 0 <= ny < len(ys) and 0 <= nx < len(xs) and free[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    queue.append((ny, nx))
        areas.append(count * spacing ** 2)
    def area(point):
        ix = min(len(xs) - 1, max(0, int((point[0] - xmin) / spacing)))
        iy = min(len(ys) - 1, max(0, int((point[1] - ymin) / spacing)))
        return areas[labels[iy, ix]]
    return area, {"frame": frame, "spacing": spacing, "areas_pixels2": areas[1:],
                  "player_component_area": area(scene.player), "geometry": "laser-only horizon partition"}


def _path_record(scene, path, hold, area):
    if path is None:
        return None
    if not replay_path(scene, path, hold):
        raise AssertionError("Returned path failed independent segment replay")
    return {"actions": list(path.actions), "points": path_points(scene, path, hold),
            "arrival_frames": len(path.actions) * hold, "endpoint": path.point,
            "min_model_clearance": path.clearance if np.isfinite(path.clearance) else None,
            "endpoint_laser_component_area": area(path.point)}


def run_case(case, beam_widths, repeats, map_path=None):
    scene, horizon = case["scene"], case["horizon"]
    start = time.perf_counter()
    events = {0., float(horizon)} | {l.on for l in scene.lasers if l.on <= horizon}
    partition_frame = max(events, key=lambda t: (sum(l.on <= t < l.off for l in scene.lasers), -t)) if scene.lasers else horizon
    area, regions = laser_regions(scene, partition_frame)
    partition_seconds = time.perf_counter() - start
    reports, results = [], []
    for width in (None, *beam_widths):
        timings = []
        for _ in range(repeats):
            result = search(scene, horizon, beam_width=width, goal=case.get("goal"),
                            position_score=None if case.get("goal") else area)
            timings.append(result.seconds * 1000)
        best = max(result.frontier, key=lambda p: (area(p.point), min(p.clearance, 24.)), default=None)
        reports.append({"method": result.method, "beam_width": width,
                        "initial_state_valid": bool(result.layers[0]),
                        "reached_frames": result.reached_frames, "horizon": horizon,
                        "layer_reachable_states": [len(layer) for layer in result.layers],
                        "edges_tested": result.edges_tested, "pruned_states": result.pruned_states,
                        "time_ms_samples": timings, "time_ms_median": statistics.median(timings),
                        "goal_arrival": _path_record(scene, result.goal_path, result.hold_frames, area),
                        "frontier_path": _path_record(scene, best, result.hold_frames, area),
                        "frontier_points": [p.point for p in result.frontier]})
        results.append(result)
    if "expect_arrival" in case and bool(results[0].goal_path) != case["expect_arrival"]:
        raise AssertionError("Reference micro-case expectation failed: " + case["name"])
    map_description = None
    if map_path is not None:
        spacing = scene.speed * 4
        xmin, xmax, ymin, ymax = scene.bounds
        low = np.ceil((np.array((xmin, ymin)) - scene.player) / spacing - 1e-5).astype(int)
        high = np.floor((np.array((xmax, ymax)) - scene.player) / spacing + 1e-5).astype(int)
        shape = (horizon // 4 + 1, int(high[1] - low[1] + 1), int(high[0] - low[0] + 1))
        arrays = {"frames": np.arange(0, horizon + 1, 4), "spacing_pixels": np.array(spacing),
                  "origin_pixels": np.array(scene.player) + low * spacing}
        for width, result in zip((None, *beam_widths), results):
            name = "reference" if width is None else f"beam_{width}"
            reachable = np.zeros(shape, dtype=np.uint8)
            for layer, entries in enumerate(result.layers):
                for x, y in entries:
                    reachable[layer, y - low[1], x - low[0]] = 1
            arrays[name + "_reachable"] = reachable
            arrays[name + "_reached_frames"] = np.array(result.reached_frames)
        np.savez_compressed(map_path, **arrays)
        map_description = {"file": map_path.name, "shape_t_y_x": shape,
                           "encoding": "1=model-reachable; 0=unmarked, not proof of danger",
                           "spacing_pixels": spacing, "origin_pixels": arrays['origin_pixels'].tolist()}
    return {"name": case["name"], "scene": asdict(scene), "goal": asdict(case["goal"]) if case.get("goal") else None,
            "metadata": case.get("metadata", {}), "expect_arrival": case.get("expect_arrival"),
            "partition_seconds": partition_seconds, "future_laser_regions": regions,
            "reachability_file": map_description, "searches": reports}


def trace_cases(path, states, horizon):
    """Read selected fresh decisions, aligning metadata to the pre-action state."""
    requested = set(states)
    found, previous = {}, {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        header = None
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if header is None:
                header = row
            if row.get("event") != "transition":
                continue
            key = row["episode"], row["step"]
            if key in requested:
                if row.get("commitment_active"):
                    raise ValueError("Select a fresh decision, not a partially held action: " + str(key))
                if header.get("hold_frames") != 4 or header.get("schema") != "touhourl-static-laser-state-v1":
                    raise ValueError("Unsupported trace timing/schema")
                scene, metadata = scene_from_observation(row["observation"])
                prior = previous.get(row["episode"])
                aligned = bool(prior and prior["step"] == row["step"] - 1
                               and prior["next_observation"] == row["observation"])
                info = prior["info"] if aligned else {}
                metadata.update(episode=key[0], step=key[1], metadata_aligned=aligned,
                                start_frame=row["info"]["elapsed_frames"] - row["info"]["delta_frames"],
                                prior_observed_bullet_count=info.get("bullet_count"),
                                prior_dropped_bullets=info.get("dropped_bullets"),
                                prior_radius_fallback_count=info.get("radius_fallback_count"),
                                prior_action=prior.get("action") if aligned else None,
                                source_model_sha256=header.get("model_sha256"))
                found[key] = {"name": f"trace-ep{key[0]}-step{key[1]}", "scene": scene,
                              "horizon": horizon, "metadata": metadata}
            previous[row["episode"]] = row
    if requested - found.keys():
        raise ValueError("Missing trace states: " + str(sorted(requested - found.keys())))
    return [found[k] for k in states], digest.hexdigest()


def write_svg(report, target):
    """Two exact-geometry views, with time-stamped route vertices."""
    scene = report["scene"]
    xmin, xmax, ymin, ymax = scene["bounds"]
    width, height = xmax - xmin, ymax - ymin
    scale = min(6., 500 / max(width, height))
    panel_width, panel_height = width * scale + 40, height * scale + 72
    horizon = report["searches"][0]["horizon"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width*2}" height="{panel_height}" viewBox="0 0 {panel_width*2} {panel_height}">',
             '<rect width="100%" height="100%" fill="white"/>']
    for panel, frame in enumerate((0, horizon)):
        tx, ty = panel * panel_width + 20, 42
        parts.append(f'<text x="{tx}" y="22" font-family="sans-serif" font-size="10">t={frame}: reference blue; beam green</text>')
        parts.append(f'<defs><clipPath id="p{panel}"><rect x="{xmin}" y="{ymin}" width="{width}" height="{height}"/></clipPath></defs>')
        parts.append(f'<g transform="translate({tx},{ty}) scale({scale}) translate({-xmin},{-ymin})">')
        parts.append(f'<rect x="{xmin}" y="{ymin}" width="{width}" height="{height}" fill="#fafafa" stroke="#444"/>')
        parts.append(f'<g clip-path="url(#p{panel})">')
        for beam in scene["lasers"]:
            active = beam["on"] <= frame < beam["off"]
            if frame >= beam["off"]:
                continue
            color = '#9467bd' if active else '#e2a435'
            stroke_width = beam['width'] if active else 1.
            parts.append(f'<line x1="{beam["ax"]}" y1="{beam["ay"]}" x2="{beam["bx"]}" y2="{beam["by"]}" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" opacity="0.5"/>')
        for circle in scene["circles"]:
            x, y = circle["x"] + circle["vx"] * frame, circle["y"] + circle["vy"] * frame
            parts.append(f'<circle cx="{x}" cy="{y}" r="{circle["radius"]}" fill="#df7474" opacity="0.55"/>')
        if panel:
            for x, y in report["searches"][0]["frontier_points"]:
                parts.append(f'<circle cx="{x}" cy="{y}" r="1.4" fill="#aaa"/>')
        for index in (0, len(report["searches"]) - 1):
            method = report["searches"][index]
            route = method["goal_arrival"] if report["goal"] else method["frontier_path"]
            if route is None:
                continue
            color = '#2672bd' if index == 0 else '#27834a'
            coordinates = ' '.join(f'{x},{y}' for x, y in route['points'])
            parts.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.8"/>')
            for step, (x, y) in enumerate(route['points']):
                parts.append(f'<circle cx="{x}" cy="{y}" r="1.5" fill="{color}"><title>t={step*4}</title></circle>')
        x, y = scene["player"]
        parts.append(f'<circle cx="{x}" cy="{y}" r="{scene["player_radius"]}" fill="black"/>')
        parts.append('</g></g>')
    parts.append('</svg>')
    target.write_text('\n'.join(parts), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--state', action='append', default=[], help='Fresh EPISODE:STEP; repeatable')
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--beam-widths', nargs='+', type=int, default=[16, 64])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('runs/planning-probe'))
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 20 or bool(args.trace) != bool(args.state):
        parser.error('Use --trace with --state; repeats must be 1..20')
    if args.horizon <= 0 or args.horizon > 256 or args.horizon % 4 or any(w < 5 for w in args.beam_widths):
        parser.error('Horizon must be a multiple of four in 4..256; beam widths must be >=5')
    try:
        states = [tuple(map(int, value.split(':'))) for value in args.state]
        if any(len(key) != 2 or min(key) < 0 for key in states) or len(states) != len(set(states)):
            raise ValueError()
    except ValueError:
        parser.error('States must be unique nonnegative EPISODE:STEP pairs')
    output = args.output
    index = 0
    while True:
        try:
            output.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            index += 1
            output = args.output.with_name(f'{args.output.name}-{index:03d}')
    output = output.resolve()
    transcript = output / 'planning.console.log'
    def announce(message):
        print(message, flush=True)
        with transcript.open('a', encoding='utf-8') as stream:
            stream.write(message + '\n')
    announce('Output: ' + str(output))
    try:
        cases = micro_cases()
        source = None
        if args.trace:
            extra, digest = trace_cases(args.trace, states, args.horizon)
            cases.extend(extra)
            source = {"path": str(args.trace.resolve()), "sha256": digest}
        report = {"model_contract": MODEL_CONTRACT, "planner_contract": PLANNER_CONTRACT,
                  "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                  "source": source, "configuration": {"beam_widths": args.beam_widths, "repeats": args.repeats,
                  "trace_horizon": args.horizon, "hold_frames": 4, "native_action_ids": [0, 1, 3, 5, 7]},
                  "limitations": LIMITATIONS, "cases": []}
        for case in cases:
            result = run_case(case, args.beam_widths, args.repeats, output / (case['name'] + '.npz'))
            write_svg(result, output / (case['name'] + '.svg'))
            report['cases'].append(result)
            (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            compact = [(r['beam_width'], r['reached_frames'], round(r['time_ms_median'], 2),
                        r['frontier_path']['endpoint_laser_component_area'] if r['frontier_path'] else None)
                       for r in result['searches']]
            announce(case['name'] + ': (width, frames, median_ms, room_area)=' + str(compact))
        announce('Report: ' + str(output / 'report.json'))
        announce('Console: ' + str(transcript))
    except BaseException:
        announce(traceback.format_exc())
        raise
    return output


if __name__ == '__main__':
    main()
