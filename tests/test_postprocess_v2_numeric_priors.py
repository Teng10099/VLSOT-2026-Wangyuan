"""CPU-only provenance, no-target, geometry and complete-candidate checks."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import postprocess_v2_numeric_priors as post


class NumericPriorPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        self.rotation = np.array([[.001, -1.0002, .02], [-.01, .025, -.9998], [1.0001, .001, .004]])
        self.translation = np.array([.5, -.2, 3.])
        K = np.array([[1000., 21., 480.], [0., 950., 272.], [0., 0., 1.]])
        size = np.array([960., 544.])
        manifest = {"schema_version": 2, "inference_version": post.infer.INFERENCE_VERSION,
                    "status": "complete_format_valid_quality_unverified", "diagnostic_only": False,
                    "official_ten_sequence_format": True, "frames_requested_per_clip": 30,
                    "target_encoding": post.infer.ENCODING, "label_encoding": post.infer.ENCODING,
                    "ground_truth_used_for_predictions": False, "world_transform_applied": False,
                    "height_offset_applied": False, "input_size_wh": [960, 544], "seed_per_video": 444,
                    "checkpoint": "/server/model.pth", "checkpoint_sha256": "a" * 64,
                    "checkpoint_epoch": 5, "state_dict_load": "strict_success", "config_sha256": "b" * 64,
                    "video_ids": list(post.infer.OFFICIAL_IDS),
                    "clips": [{"video_id": item} for item in post.infer.OFFICIAL_IDS],
                    "result_file_sha256": {}, "coordinate_note": post.infer.COORDINATE_NOTE}
        for clip_id in post.infer.OFFICIAL_IDS:
            directory = self.source / clip_id
            directory.mkdir()
            frames = []
            for index in range(1, 31):
                native = np.array([25. + index * .01, 6., -.5])
                camera = self.rotation @ native + self.translation
                projected = K @ camera
                xy = projected[:2] / projected[2] / size
                heads = {"pred_boxes": np.array([[np.r_[xy, [0., 0., 0., 0.]]]]),
                         "pred_depth": np.array([[[camera[2], 0.]]]),
                         "pred_3d_dim": np.array([[[1.5, 2., 4.]]]),
                         "target_encoding": post.infer.ENCODING}
                frame = {"frame_index": index, "image_name": "{:06d}.jpg".format(index),
                         "image_sizes": size, "calibs": np.column_stack((K, np.zeros(3))),
                         "lidar_to_camera_R": self.rotation, "lidar_to_camera_t": self.translation}
                frame.update(post.infer.decode_prediction(heads, frame))
                frame.update({key: value for key, value in heads.items() if key != "target_encoding"})
                frames.append(post.infer._jsonable(frame))
            diagnostic = {key: manifest[key] for key in ("checkpoint", "checkpoint_epoch", "seed_per_video",
                          "state_dict_load", "target_encoding", "label_encoding")}
            diagnostic.update(video_id=clip_id, sequence_id="0000", frames=frames,
                              input_text="Initially, a van is 20 meters away. It measures 1.1 meters in height, "
                                         "5.5 meters in length, and 1.6 meters in width. Later it moves away.")
            post.infer._write_json(directory / "diagnostics.json", diagnostic)
            text = "\n".join(post.infer._format_row(i, np.asarray(frame["values_lwhxyz"], dtype=np.float64))
                             for i, frame in enumerate(frames, 1)) + "\n"
            manifest["result_file_sha256"][clip_id] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        post.infer._write_json(self.source / "inference_manifest.json", manifest)

    def read(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    def change_manifest(self, **changes):
        path = self.source / "inference_manifest.json"
        value = self.read(path)
        value.update(changes)
        post.infer._write_json(path, value)

    def change_diagnostic(self, change):
        path = self.source / post.infer.OFFICIAL_IDS[0] / "diagnostics.json"
        value = self.read(path)
        change(value)
        post.infer._write_json(path, value)

    def test_range_dims_complete_zip_and_geometry_and_raw_preservation(self):
        before = {path: post.infer.sha256(path) for path in self.source.rglob("*.json")}
        result = post.postprocess(self.source, self.root / "corrected", "initial_range_and_dimensions")
        self.assertEqual(result["frame_count"], 300)
        self.assertFalse(result["officially_scored"])
        self.assertEqual(result["checkpoint_sha256"], "a" * 64)
        self.assertEqual(result["source_manifest_sha256"], before[self.source / "inference_manifest.json"])
        self.assertEqual(before, {path: post.infer.sha256(path) for path in self.source.rglob("*.json")})
        self.assertTrue(all(item["range_applied_frames"] == 30 for item in result["clips"]))
        frame = self.read(self.root / "corrected" / post.infer.OFFICIAL_IDS[0] / "diagnostics.json")["frames"][0]
        raw = frame["raw_decoded"]
        native = np.asarray(frame["label_position_native_xyz"])
        self.assertAlmostEqual(np.linalg.norm(native), 20.)
        np.testing.assert_allclose(native / np.linalg.norm(native),
                                   np.asarray(raw["label_position_native_xyz"]) / np.linalg.norm(raw["label_position_native_xyz"]))
        np.testing.assert_allclose(frame["values_lwhxyz"], np.r_[[5.5, 1.6, 1.1], native])
        np.testing.assert_allclose(frame["camera_center_xyz"], self.rotation @ native + self.translation)
        self.assertAlmostEqual(frame["camera_z"], frame["camera_center_xyz"][2])
        np.testing.assert_allclose(frame["pred_3d_dim"], [[[1.5, 2., 4.]]])
        self.assertTrue(frame["camera_metrics_available"])
        with zipfile.ZipFile(result["zip"]) as archive:
            self.assertEqual(archive.namelist(), ["result/{}.txt".format(item) for item in post.infer.OFFICIAL_IDS])
            self.assertIsNone(archive.testzip())
            for clip in post.infer.OFFICIAL_IDS:
                post.infer._validate_text(archive.read("result/{}.txt".format(clip)).decode("utf-8"), clip)

    def test_dims_only_keeps_native_and_camera_geometry(self):
        result = post.postprocess(self.source, self.root / "dims", "dims_only")
        self.assertTrue(all(item["range_applied_frames"] == 0 for item in result["clips"]))
        frame = self.read(self.root / "dims" / post.infer.OFFICIAL_IDS[0] / "diagnostics.json")["frames"][0]
        np.testing.assert_allclose(frame["label_position_native_xyz"], frame["raw_decoded"]["label_position_native_xyz"])
        np.testing.assert_allclose(frame["camera_center_xyz"], frame["raw_decoded"]["camera_center_xyz"])
        self.assertEqual(frame["dimensions_lwh"], [5.5, 1.6, 1.1])

    def test_ambiguous_missing_priors_are_safe_fallback(self):
        self.change_diagnostic(lambda item: item.update(input_text="Initially a van is 20 meters away and 30 meters away."))
        result = post.postprocess(self.source, self.root / "fallback", "initial_range_and_dimensions")
        self.assertEqual(result["clips"][0]["range_applied_frames"], 0)
        self.assertEqual(result["clips"][0]["dimensions_applied_frames"], 0)

    def test_checkpoint_epoch_or_seed_mismatch_is_rejected_before_output(self):
        for changes in ({"checkpoint": "/other/model.pth"}, {"checkpoint_epoch": 4},
                        {"seed_per_video": 1}, {"checkpoint_sha256": "c" * 64}):
            with self.subTest(changes=changes):
                original = self.read(self.source / post.infer.OFFICIAL_IDS[0] / "diagnostics.json")
                self.change_diagnostic(lambda item: item.update(changes))
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    post.postprocess(self.source, self.root / "bad", "dims_only")
                self.assertFalse((self.root / "bad").exists())
                post.infer._write_json(self.source / post.infer.OFFICIAL_IDS[0] / "diagnostics.json", original)

    def test_partial_or_unordered_source_is_rejected(self):
        self.change_manifest(status="running")
        with self.assertRaisesRegex(ValueError, "completed"):
            post.validate_source(self.source)
        self.change_manifest(status="complete_format_valid_quality_unverified")
        self.change_diagnostic(lambda item: item["frames"].reverse())
        with self.assertRaisesRegex(ValueError, "30 ordered"):
            post.validate_source(self.source)

    def test_target_like_fields_refused_recursively(self):
        for key in ("target", "gt_native_xyz", "ground_truth", "native_xyz_target", "annotations"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Target-like"):
                    post.reject_target_fields({"frames": [{key: [1., 2., 3.]}]})
        self.change_diagnostic(lambda item: item["frames"][0].update(target={"xyz": [1, 2, 3]}))
        with self.assertRaisesRegex(ValueError, "Target-like"):
            post.postprocess(self.source, self.root / "bad", "dims_only")
        self.assertFalse((self.root / "bad").exists())

    def test_raw_head_decoded_and_manifest_commitment_consistency(self):
        self.change_diagnostic(lambda item: item["frames"][0]["values_lwhxyz"].__setitem__(0, 999.))
        with self.assertRaisesRegex(ValueError, "inconsistency"):
            post.validate_source(self.source)

    def test_manifest_hashes_and_unexpected_diagnostics_refused(self):
        manifest = self.read(self.source / "inference_manifest.json")
        manifest["result_file_sha256"][post.infer.OFFICIAL_IDS[0]] = "c" * 64
        post.infer._write_json(self.source / "inference_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "committed TXT"):
            post.validate_source(self.source)
        extra = self.source / "999999"
        extra.mkdir()
        (extra / "diagnostics.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            post.validate_source(self.source)

    def test_source_related_or_existing_output_refused(self):
        for output in (self.source, self.source / "nested", self.root):
            with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                post.postprocess(self.source, output, "dims_only")
        other = self.root / "existing"
        other.mkdir()
        with self.assertRaises(FileExistsError):
            post.postprocess(self.source, other, "dims_only")

    def test_json_duplicate_and_nonfinite_rejected_and_python38_syntax(self):
        path = self.root / "invalid.json"
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                post._read_json(path)
        ast.parse(Path(post.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))


if __name__ == "__main__":
    unittest.main()
