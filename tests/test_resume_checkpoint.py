import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.pipeline import (
    _append_l1l2_checkpoint,
    _read_l1l2_checkpoint,
    _resume_signature,
    _write_resume_manifest,
)


class ResumeCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.file = self.input_dir / "one.txt"
        self.file.write_text("first content", encoding="utf-8")
        self.output = self.root / "results.csv"
        self.config = {
            "project": {"name": "Demo"},
            "project_context": "Demo project",
            "model": "test-model",
            "api_key": "must-not-be-persisted",
            "taxonomy": {"domains": ["A"]},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoint_round_trip_and_ignores_incomplete_line(self):
        checkpoint = self.root / "resume.jsonl"
        l1 = {"filename": "one.txt", "file_path": str(self.file)}
        l2 = {"domain": "A", "keywords": ["first", "content"]}
        _append_l1l2_checkpoint(checkpoint, self.file, l1, l2)
        with checkpoint.open("a", encoding="utf-8") as stream:
            stream.write('{"partial":')

        records = _read_l1l2_checkpoint(checkpoint)
        self.assertEqual(records[str(self.file.resolve())]["l1"], l1)
        self.assertEqual(records[str(self.file.resolve())]["l2"], l2)

    def test_checkpoint_is_invalidated_when_input_changes(self):
        checkpoint = self.root / "resume.jsonl"
        _append_l1l2_checkpoint(checkpoint, self.file, {"filename": "one.txt"}, {"domain": "A"})
        self.file.write_text("changed content", encoding="utf-8")
        self.assertEqual(_read_l1l2_checkpoint(checkpoint), {})

    def test_manifest_signature_binds_inputs_and_settings_but_not_api_key(self):
        files = [self.file]
        initial = _resume_signature(self.input_dir, self.output, files, self.config)
        rotated_key = {**self.config, "api_key": "different-secret"}
        self.assertEqual(initial, _resume_signature(self.input_dir, self.output, files, rotated_key))

        changed_config = {**self.config, "model": "other-model"}
        self.assertNotEqual(initial, _resume_signature(self.input_dir, self.output, files, changed_config))
        changed_taxonomy = {**self.config, "taxonomy": {"domains": ["B"]}}
        self.assertNotEqual(initial, _resume_signature(self.input_dir, self.output, files, changed_taxonomy))

        self.file.write_text("changed content", encoding="utf-8")
        self.assertNotEqual(initial, _resume_signature(self.input_dir, self.output, files, self.config))

    def test_pipeline_reuses_phase1_cache_and_appends_only_unfinished_rows(self):
        from core import pipeline

        two = self.input_dir / "two.txt"
        two.write_text("second content", encoding="utf-8")
        files = [self.file, two]
        config = {**self.config, "base_url": "https://api.example.test/v1", "api_timeout": 5,
                  "temperature": 0, "delay": 0, "layer2_settings": {}}
        signature = _resume_signature(self.input_dir, self.output, files, config)
        _write_resume_manifest(Path(str(self.output) + ".resume.json"), signature,
                               self.input_dir, self.output, files)
        completed_row = {field: "" for field in pipeline.FIELDNAMES}
        completed_row.update(file_path=str(two), filename="two.txt")
        import csv
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=pipeline.FIELDNAMES)
            writer.writeheader()
            writer.writerow(completed_row)
        _append_l1l2_checkpoint(Path(str(self.output) + ".resume.jsonl"), self.file,
                               {"filename": "one.txt", "file_path": str(self.file), "format": "txt"},
                               {"domain": "A", "lifecycle": "Planning", "asset_type": "Document",
                                "information_type": "Document", "keywords": "one", "short_summary": "one"})
        signature = _resume_signature(self.input_dir, self.output, files, config)
        _write_resume_manifest(Path(str(self.output) + ".resume.json"), signature,
                               self.input_dir, self.output, files)

        called = []
        fake_l1 = {"filename": "two.txt", "file_path": str(two), "format": "txt"}
        fake_l2 = {"domain": "B", "lifecycle": "Planning", "asset_type": "Document",
                   "information_type": "Document", "keywords": "two", "short_summary": "two"}
        fake_row = {field: "" for field in pipeline.FIELDNAMES}
        fake_row.update(file_path=str(self.file), filename="one.txt", review_priority="Low")

        with patch.object(pipeline, "OpenAI") as mock_client, \
             patch.object(pipeline, "extract_project_intelligence", return_value={"sources": []}), \
             patch.object(pipeline, "_run_l1l2", side_effect=lambda fp, *args, **kwargs: called.append(str(fp)) or (fake_l1, fake_l2)), \
             patch.object(pipeline, "_run_l3l4", return_value=fake_row), \
             patch.object(pipeline, "_agent_phase", return_value=0):
            mock_client.return_value = object()
            pipeline.run(self.input_dir, self.output, config, parallel=1, resume=True)

        self.assertEqual(called, [])
        rows = list(__import__("csv").DictReader(self.output.open(encoding="utf-8")))
        self.assertEqual([row["file_path"] for row in rows].count(str(two)), 1)
        self.assertEqual([row["file_path"] for row in rows].count(str(self.file)), 1)

    def test_manifest_has_inventory_but_never_credentials(self):
        manifest_path = self.root / "resume.json"
        signature = _resume_signature(self.input_dir, self.output, [self.file], self.config)
        _write_resume_manifest(manifest_path, signature, self.input_dir, self.output, [self.file])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"], [str(self.file.resolve())])
        self.assertNotIn("api_key", manifest)
        self.assertNotIn("must-not-be-persisted", manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
