"""Metadata-only provenance checks; no model, target labels or image files."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "tools" / "audit_training_inputs.py"
SPEC = importlib.util.spec_from_file_location("audit_training_inputs", str(MODULE))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class TrainingInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.videos = self.root / "video"
        self.videos.mkdir()

    def split(self, name, ids):
        path = self.root / name
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        return path

    def clip(self, clip_id, scene, frames=None, numeric_id=False):
        folder = self.videos / scene if scene else self.videos
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (clip_id + ".json")
        frames = frames or [clip_id + "_{}.jpg".format(i) for i in range(30)]
        record = {"videoID": int(clip_id) if numeric_id else clip_id}
        for index, frame in enumerate(frames):
            # Neither calibration files, images nor numerical targets exist.
            record["frame{}".format(index)] = [True, "old/calib.json", frame]
        path.write_text(json.dumps([record]), encoding="utf-8")
        return path

    def test_independent_known_scenes_and_clean_official_test_are_ready(self):
        self.clip("000001", "sequence_0001", numeric_id=True)
        self.clip("000002", "sequence_0002")
        self.clip("000003", "sequence_0003")
        report = audit.audit_inputs(
            self.split("train.txt", ["000001.json", "000002"]), self.videos,
            self.split("official_test.txt", ["000003"]))
        self.assertEqual(report["training"]["unique_frame_count"], 60)
        self.assertEqual(report["safe_validation"]["group_count"], 2)
        self.assertTrue(report["safe_validation"]["ready_for_independent_validation"])
        self.assertEqual(report["safe_validation"]["status"], "groups_available")
        self.assertEqual(report["clip_index"][0]["clip_id"], "000001")
        self.assertEqual(report["rules"]["fields_used"], ["videoID", "frame0..29[2]"])

    def test_transitive_shared_frames_form_one_group_and_40_images(self):
        for clip_id, start in [("000001", 0), ("000002", 5), ("000003", 10)]:
            self.clip(clip_id, "sequence_0042", ["{}.jpg".format(i) for i in range(start, start + 30)])
        report = audit.audit_inputs(self.split("train.txt", ["000001", "000002", "000003"]), self.videos)
        summary = report["training"]
        self.assertEqual(summary["unique_frame_count"], 40)
        self.assertEqual([c["clip_count"] for c in summary["shared_frame_components"]], [3])
        self.assertEqual(summary["scenes"][0]["scene"], "sequence_0042")
        self.assertEqual(summary["scenes"][0]["unique_frame_count"], 40)
        self.assertFalse(report["safe_validation"]["can_split_by_group"])
        self.assertIn("Do not use a random frame or clip split", report["safe_validation"]["reason"])

    def test_same_recording_is_one_leakage_group_even_without_shared_images(self):
        self.clip("000001", "sequence_0001")
        self.clip("000002", "sequence_0001")
        report = audit.audit_inputs(self.split("train.txt", ["000001", "000002"]), self.videos)
        self.assertEqual(len(report["training"]["shared_frame_components"]), 2)
        self.assertEqual(len(report["training"]["leakage_groups"]), 1)
        self.assertFalse(report["safe_validation"]["can_split_by_group"])

    def test_cross_recording_shared_basename_merges_groups_and_detects_test_overlap(self):
        first = ["old/images/shared.jpg"] + ["first_{}.jpg".format(i) for i in range(29)]
        second = [r"C:\images\shared.jpg"] + ["second_{}.jpg".format(i) for i in range(29)]
        self.clip("000001", "sequence_0001", first)
        self.clip("000002", "sequence_0002", second)
        self.clip("000003", "sequence_0003", ["shared.jpg"] + ["test_{}.jpg".format(i) for i in range(29)])
        report = audit.audit_inputs(
            self.split("train.txt", ["000001", "000002"]), self.videos,
            self.split("test.txt", ["000003"]))
        self.assertEqual(report["safe_validation"]["group_count"], 1)
        self.assertEqual(report["official_test_overlap"]["frame_overlap"], ["shared.jpg"])
        self.assertEqual(report["official_test_overlap"]["train_clip_ids"], ["000001", "000002"])
        self.assertEqual(report["official_test_overlap"]["test_clip_ids"], ["000003"])
        self.assertFalse(report["safe_validation"]["official_test_clean"])

    def test_official_test_id_overlap_is_reported(self):
        self.clip("000001", "sequence_0001")
        self.clip("000002", "sequence_0002")
        report = audit.audit_inputs(
            self.split("train.txt", ["000001", "000002"]), self.videos,
            self.split("test.txt", ["000001"]))
        self.assertEqual(report["official_test_overlap"]["id_overlap"], ["000001"])
        self.assertEqual(report["safe_validation"]["status"], "official_test_overlap")
        self.assertFalse(report["safe_validation"]["ready_for_independent_validation"])

    def test_unknown_recording_and_unchecked_test_never_claim_independent_validation(self):
        self.clip("000001", None)
        self.clip("000002", "sequence_0002")
        split = self.split("train.txt", ["000001", "000002"])
        report = audit.audit_inputs(split, self.videos)
        self.assertTrue(report["safe_validation"]["can_split_by_group"])
        self.assertEqual(report["training"]["unknown_scene_clip_ids"], ["000001"])
        self.assertIsNone(report["safe_validation"]["official_test_clean"])
        self.assertFalse(report["safe_validation"]["ready_for_independent_validation"])
        self.assertEqual(report, audit.audit_inputs(split, self.videos))

    def test_missing_ambiguous_duplicate_ids_and_frames_are_rejected(self):
        split = self.split("train.txt", ["000001"])
        with self.assertRaisesRegex(ValueError, "missing"):
            audit.audit_inputs(split, self.videos)
        self.clip("000001", "sequence_0001")
        duplicate = self.clip("000001", "sequence_0002")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            audit.audit_inputs(split, self.videos)
        duplicate.unlink()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.audit_inputs(self.split("duplicate.txt", ["000001", "000001.json"]), self.videos)
        self.clip("000001", "sequence_0001", ["same.jpg"] * 30)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit.audit_inputs(split, self.videos)

    def test_mismatching_video_id_and_incomplete_clip_are_rejected(self):
        path = self.clip("000001", "sequence_0001")
        split = self.split("train.txt", ["000001"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw[0]["videoID"] = "000002"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "videoID"):
            audit.audit_inputs(split, self.videos)
        raw[0]["videoID"] = "000001"
        del raw[0]["frame29"]
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing"):
            audit.audit_inputs(split, self.videos)

    def test_cli_preserves_inputs_and_existing_report(self):
        path = self.clip("000001", "sequence_0001")
        split = self.split("train.txt", ["000001"])
        output = self.root / "report.json"
        before = (path.read_bytes(), split.read_bytes())
        arguments = ["--train-split", str(split), "--video-root", str(self.videos), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(audit.main(arguments), 0)
            report_bytes = output.read_bytes()
            self.assertEqual(audit.main(arguments), 2)
        self.assertEqual(output.read_bytes(), report_bytes)
        self.assertEqual((path.read_bytes(), split.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
