"""Explicit task provenance, per-task splits and independent joint cloning."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.demonstrations import deduplicate_episodes
from agent.zone_demonstrations import (load_zone_demonstrations,
                                       load_multitask_zone_demonstrations, split_zone_tasks)
from scripts.evaluate_control import evaluation_identity
from tests.test_demonstrations import write_trace
from tests.test_ppo_backend import AVAILABLE
from tests.test_zone_demonstrations import zone_rows


def source(directory, name, offset=0., width=24):
    rows = zone_rows(width)
    rows[0]["task"] = name
    for row in rows:
        if row["event"] == "transition":
            for key in ("observation", "next_observation"):
                # Lives distinguish complete trajectories without affecting rule ranking.
                row[key]["player"][2] = offset + row["episode"] * .01
    path = Path(directory) / (name + ".jsonl")
    write_trace(path, rows)
    return path


class MultitaskZoneTests(unittest.TestCase):
    def test_explicit_loader_retains_tasks_and_legacy_loader_stays_strict(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = [source(directory, "alpha"), source(directory, "beta", .1)]
            with self.assertRaises(ValueError):
                load_zone_demonstrations(paths)
            episodes, provenance = load_multitask_zone_demonstrations(paths)
            self.assertEqual(provenance["identity"]["tasks"], ["alpha", "beta"])
            self.assertNotIn("task", provenance["identity"])
            self.assertEqual([e["task"] for e in episodes], ["alpha", "alpha", "beta", "beta"])
            self.assertEqual([s["task"] for s in provenance["sources"]], ["alpha", "beta"])
            self.assertEqual(provenance["dataset_contract"], "attack-zone-multitask-v1")
            self.assertEqual(provenance["identity"]["decision_contract"], "attack-zone-held-v1")

    def test_incompatible_settings_single_task_and_duplicate_files_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            a, b = source(directory, "alpha"), source(directory, "beta", .1)
            for paths in ([a], [a, b, a], [a, source(directory, "narrow", .2, 6)]):
                with self.subTest(paths=paths), self.assertRaises(ValueError):
                    load_multitask_zone_demonstrations(paths)

    def test_multi_loader_does_not_bypass_original_telemetry_validation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            a, b = source(directory, "alpha"), source(directory, "beta", .1)
            rows = [json.loads(line) for line in b.read_text().splitlines()]
            del rows[2]["movement_diagnostic"]
            b.unlink()
            write_trace(b, rows)
            with self.assertRaises(ValueError):
                load_multitask_zone_demonstrations([a, b])

    def test_split_is_deterministic_disjoint_and_covers_each_task(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            episodes, provenance = load_multitask_zone_demonstrations(
                [source(directory, "alpha"), source(directory, "beta", .1)])
            episodes, duplicates = deduplicate_episodes(episodes)
            self.assertFalse(duplicates)
            train, validation, splits = split_zone_tasks(episodes, provenance["identity"]["tasks"], 0)
            self.assertEqual({e["task"] for e in train}, {"alpha", "beta"})
            self.assertEqual({e["task"] for e in validation}, {"alpha", "beta"})
            self.assertFalse({e["id"] for e in train} & {e["id"] for e in validation})
            self.assertEqual(splits, split_zone_tasks(episodes, provenance["identity"]["tasks"], 0)[2])
            self.assertTrue(all(s["train_samples"] == 2 and s["validation_samples"] == 2
                                for s in splits.values()))
            episodes[1]["observations"] = deepcopy(episodes[0]["observations"])
            unique, duplicates = deduplicate_episodes(episodes)
            self.assertEqual(len(duplicates), 1)
            with self.assertRaises(ValueError):
                split_zone_tasks(unique, provenance["identity"]["tasks"])

    def test_seen_tasks_and_unseen_transfer_have_distinct_evaluation_rules(self):
        saved = {"initialization": {"identity": {"tasks": ["alpha", "beta"]}}}
        before = deepcopy(saved)
        for task in ("alpha", "beta"):
            identity = evaluation_identity(saved, task)
            self.assertFalse(identity["task_transfer"])
            self.assertIsNone(identity["training_task"])
            self.assertEqual(identity["training_tasks"], ["alpha", "beta"])
        with self.assertRaises(ValueError):
            evaluation_identity(saved, "new")
        moved = evaluation_identity(saved, "new", allow_task_transfer=True, transfer_reason="new spell")
        self.assertTrue(moved["task_transfer"])
        with self.assertRaises(ValueError):
            evaluation_identity(saved, "alpha", allow_task_transfer=True, transfer_reason="already seen")
        for tasks in (["alpha"], ["alpha", "alpha"], "alpha", ["alpha", ""]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                evaluation_identity({"initialization": {"identity": {"tasks": tasks}}}, "alpha")
        self.assertEqual(saved, before)

    @unittest.skipUnless(AVAILABLE, "Install training dependencies")
    def test_joint_cli_publishes_original_tasks_without_changing_sources_or_training_guards(self):
        from scripts.clone import main
        from scripts.train_held import main as train_held
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = [source(directory, "alpha"), source(directory, "beta", .1)]
            original = [p.read_bytes() for p in paths]
            run = Path(directory) / "joint"
            args = ["clone", "--multitask-zone-demos", *map(str, paths),
                    "--epochs", "1", "--run-dir", str(run)]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                main()
            checkpoint = run / "checkpoints/update-000000"
            saved = json.loads((checkpoint / "metadata.json").read_text())
            self.assertEqual(saved["initialization"]["identity"]["tasks"], ["alpha", "beta"])
            self.assertEqual(set(saved["initialization"]["task_splits"]), {"alpha", "beta"})
            self.assertEqual(saved["config"]["feature_options"]["half_width"], 24)
            self.assertFalse(evaluation_identity(saved, "beta")["task_transfer"])
            self.assertEqual([p.read_bytes() for p in paths], original)
            self.assertEqual((saved["updates"], saved["timesteps"]), (0, 0))
            args = ["train_held", "--resume", str(checkpoint), "--task", "alpha", "--run-dir", "unused"]
            with patch("sys.argv", args), patch("scripts.train_held.connect") as connect, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_held()
            connect.assert_not_called()
