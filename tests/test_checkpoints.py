import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.checkpoints import create_run_directory, read_checkpoint, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_transient_rename_denial_is_retried_without_overwriting(self):
        from agent.checkpoints import _publish_directory
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            pending, target = Path(directory) / "pending", Path(directory) / "published"
            pending.mkdir()
            (pending / "model.zip").write_bytes(b"complete")
            original = Path.rename
            attempts = []
            def rename(path, destination):
                attempts.append(path)
                if len(attempts) < 3:
                    raise PermissionError("temporarily busy")
                return original(path, destination)
            with patch.object(Path, "rename", autospec=True, side_effect=rename), patch("agent.checkpoints.time.sleep"):
                _publish_directory(pending, target)
            self.assertEqual(len(attempts), 3)
            self.assertEqual((target / "model.zip").read_bytes(), b"complete")
            pending.mkdir()
            with self.assertRaises(FileExistsError):
                _publish_directory(pending, target)

    def test_old_reward_checkpoint_is_evaluation_only(self):
        class Learner:
            def save(self, path):
                Path(path).write_bytes(b"fake-model")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            saved = save_checkpoint(Learner(), directory,
                                    {"updates": 1, "timesteps": 6}, {}, "test-build")
            file = saved / "metadata.json"
            metadata = json.loads(file.read_text())
            metadata["reward_profile"] = "spell-v1"
            file.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_checkpoint(saved)
            self.assertEqual(read_checkpoint(saved, evaluation=True)["reward_profile"], "spell-v1")
            metadata["reward_profile"] = "unknown"
            file.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                read_checkpoint(saved, evaluation=True)

    def test_publish_load_and_duplicate_preservation(self):
        class Learner:
            def save(self, path):
                Path(path).write_bytes(b"fake-model")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = create_run_directory(Path(directory) / "run")
            other = create_run_directory(Path(directory) / "run")
            self.assertNotEqual(root, other)
            state = {"updates": 1, "timesteps": 6}
            saved = save_checkpoint(Learner(), root, state, {"capacity": 128}, "test-build")
            self.assertEqual(read_checkpoint(saved)["timesteps"], 6)
            self.assertEqual(json.loads((saved.parent / "latest.json").read_text())["checkpoint"], saved.name)
            with self.assertRaises(FileExistsError):
                save_checkpoint(Learner(), root, state, {}, "test-build")

    def test_failed_save_is_never_published(self):
        class Broken:
            def save(self, path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("save interrupted")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with self.assertRaises(RuntimeError):
                save_checkpoint(Broken(), directory, {"updates": 1}, {}, "test-build")
            self.assertFalse((Path(directory) / "checkpoints" / "update-000001").exists())

    def test_partial_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)
            (path / "metadata.json").write_text('{"complete": false}')
            with self.assertRaises(ValueError):
                read_checkpoint(path)

    def test_missing_model_archive_is_not_published(self):
        class Silent:
            def save(self, path):
                pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with self.assertRaises(RuntimeError):
                save_checkpoint(Silent(), directory, {"updates": 1, "timesteps": 2}, {}, "test")
            self.assertFalse((Path(directory) / "checkpoints" / "update-000001").exists())
