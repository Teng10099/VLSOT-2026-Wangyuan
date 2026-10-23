"""Submission packaging contracts, with synthetic model heads and no dataset."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("export_vlsot", str(TOOLS / "export_vlsot.py"))
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportVLSOTTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "submission"

    def sequence(self, video_id="000237", frames=30, directory=None,
                 dimensions=(1.5, 2., 4.), third_axis=-.5):
        directory = directory or self.inputs / video_id
        directory.mkdir(parents=True)
        arrays = {}
        records = []
        # Source 1280x720 -> input 1920x1200: uniform scale 1.5 plus y offset 60.
        # Intended native label: x=25, y=6, z=-.5. Dataset subtracts h/2 from y.
        normalized_xy = [25. * 1.5 / 1920., (5.25 * 1.5 + 60.) / 1200.]
        for index in range(1, frames + 1):
            heads = {
                "pred_logits": np.array([[[1., 3.]]]),
                "pred_boxes": np.array([[normalized_xy + [.1, .1, .2, .2]]]),
                "pred_3d_dim": np.array([[dimensions]], dtype=np.float64),
                "pred_depth": np.array([[[third_axis, 0.]]]),
            }
            for key, value in heads.items():
                arrays["frame_{:02d}_{}".format(index, key)] = value
            records.append({
                "frame_index": index,
                "image_name": "image_{}.jpg".format(index),
                "source_size_wh": [1280, 720],
                "input_size_wh": [1920, 1200],
                "source_from_input": [[2. / 3., 0., 0.], [0., 2. / 3., -40.]],
                # Deliberately contradictory labels: exporter must ignore them.
                "ground_truth": [999., 999., 999.],
            })
        record = {"video_id": video_id, "state_dict_load": "strict_success",
                  "checkpoint": "unavailable/on/packaging/computer.pth",
                  "frames": records}
        (directory / "diagnostics.json").write_text(json.dumps(record), encoding="utf-8")
        np.savez_compressed(directory / "raw_outputs.npz", **arrays)
        return directory

    def diagnostic(self):
        return exporter.export_submission(
            self.inputs, self.output, "legacy_native_xy",
            expected_ids=None, diagnostic_single_id="000237")

    def transforms(self, ids=("000237",)):
        document = {
            "schema_version": 1,
            "source_coordinate_system": "infrastructure_virtual_lidar",
            "target_coordinate_system": "virtual_world",
            "source_verification": "Synthetic fixture representing training-source provenance",
            "sequences": {},
        }
        for video_id in ids:
            document["sequences"][video_id] = [{
                "frame_index": index,
                "image_name": "image_{}.jpg".format(index),
                # 90-degree rotation around the Z axis, then translation.
                "rotation": [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                "translation": [[100.], [-20.], [3.]],
                "calibration_path": "/calib/virtuallidar_to_world/image_{}.json".format(index),
                "calibration_sha256": "0" * 64,
            } for index in range(1, 31)]
        path = self.root / "world_transforms.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path, document

    def world_diagnostic(self, transforms):
        return exporter.export_submission(
            self.inputs, self.output, "legacy_native_xy_to_world",
            expected_ids=None, diagnostic_single_id="000237", world_transforms=transforms)

    def test_exact_native_decode_inverse_affine_order_and_negative_third_axis(self):
        self.sequence()
        result = self.diagnostic()
        with zipfile.ZipFile(result["zip"]) as archive:
            self.assertEqual(archive.namelist(), ["result/000237.txt"])
            lines = archive.read("result/000237.txt").decode().splitlines()
        self.assertEqual(len(lines), 30)
        self.assertEqual(lines[0].split()[0], "1")
        self.assertEqual(lines[-1].split()[0], "30")
        np.testing.assert_allclose([float(x) for x in lines[0].split()[1:]],
                                   [4., 2., 1.5, 25., 6., -.5], atol=1e-9)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertFalse(manifest["world_coordinates_verified"])
        self.assertFalse(manifest["ground_truth_used_for_predictions"])
        self.assertTrue(manifest["diagnostic_only"])

    def test_official_archive_has_exactly_ten_ordered_members(self):
        for video_id in reversed(exporter.OFFICIAL_IDS):
            self.sequence(video_id)
        result = exporter.export_submission(self.inputs, self.output,
                                            "legacy_native_xy", "official")
        with zipfile.ZipFile(result["zip"]) as archive:
            self.assertEqual(archive.namelist(), [
                "result/{}.txt".format(value) for value in exporter.OFFICIAL_IDS])
            self.assertNotIn("manifest.json", archive.namelist())
        self.assertTrue(result["official_ten_sequence_format"])

    def test_refuses_partial_sequence_without_leaving_output(self):
        self.sequence(frames=29)
        with self.assertRaisesRegex(ValueError, "30"):
            self.diagnostic()
        self.assertFalse(self.output.exists())

    def test_refuses_duplicate_ids_from_recursive_inference_runs(self):
        self.sequence(directory=self.inputs / "run_a" / "sequence")
        self.sequence(directory=self.inputs / "run_b" / "sequence")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.diagnostic()

    def test_rejects_negative_dimensions_without_abs_or_clamp(self):
        self.sequence(dimensions=(-1.5, 2., 4.))
        with self.assertRaisesRegex(ValueError, "positive"):
            self.diagnostic()
        self.assertFalse(self.output.exists())

    def test_rejects_nonfinite_values(self):
        self.sequence(third_axis=float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.diagnostic()

    def test_refuses_existing_output_directory_and_preserves_zip(self):
        self.sequence()
        result = self.diagnostic()
        previous = Path(result["zip"]).read_bytes()
        with self.assertRaises(FileExistsError):
            self.diagnostic()
        self.assertEqual(previous, Path(result["zip"]).read_bytes())

    def test_official_mode_cannot_silently_accept_one_sequence(self):
        self.sequence()
        with self.assertRaisesRegex(ValueError, "missing"):
            exporter.export_submission(self.inputs, self.output,
                                        "legacy_native_xy", "official")

    def test_rejects_json_npz_pair_mismatch_and_duplicate_frame_indices(self):
        directory = self.sequence()
        path = directory / "diagnostics.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["frames"][0]["pred_depth"] = [[[500., 0.]]]
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "disagree"):
            self.diagnostic()
        record["frames"][0].pop("pred_depth")
        record["frames"][1]["frame_index"] = 1
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate frame"):
            self.diagnostic()

    def test_coordinate_policy_and_official_id_set_are_explicit(self):
        self.sequence()
        with self.assertRaisesRegex(ValueError, "Explicit"):
            exporter.export_submission(self.inputs, self.output,
                                        "projected_3d_center", "official")
        with self.assertRaisesRegex(ValueError, "fixed ten"):
            exporter.export_submission(self.inputs, self.output,
                                        "legacy_native_xy", ["000237"])
        args = exporter.parse_args(["--input-dir", str(self.inputs),
                                    "--output-dir", str(self.output),
                                    "--coordinate-policy", "legacy_native_xy",
                                    "--expected-ids", "official"])
        self.assertEqual(args.expected_ids, "official")

    def test_world_transform_rotates_then_translates_position_only_and_keeps_backup(self):
        self.sequence()
        path, _ = self.transforms()
        result = self.world_diagnostic(path)
        with zipfile.ZipFile(result["zip"]) as archive:
            world_row = archive.read("result/000237.txt").decode().splitlines()[0].split()
            self.assertEqual(archive.namelist(), ["result/000237.txt"])
        with zipfile.ZipFile(result["native_backup_zip"]) as archive:
            native_row = archive.read("result/000237.txt").decode().splitlines()[0].split()
        np.testing.assert_allclose([float(x) for x in world_row[1:]],
                                   [4., 2., 1.5, 94., 5., 2.5])
        np.testing.assert_allclose([float(x) for x in native_row[1:]],
                                   [4., 2., 1.5, 25., 6., -.5])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertTrue(manifest["world_transform_applied"])
        self.assertEqual(manifest["coordinate_system"], "virtual_world")
        self.assertEqual(manifest["native_coordinate_system"], "infrastructure_virtual_lidar")
        self.assertFalse(manifest["platform_coordinate_interpretation_confirmed_by_scoring"])
        self.assertFalse(manifest["ground_truth_used_for_predictions"])
        self.assertIn("official_documentation", manifest["world_transform_provenance"])
        self.assertIn("world_transform", manifest["sequences"][0]["frames"][0])

    def test_world_mode_requires_exactly_matching_per_image_calibration(self):
        self.sequence()
        path, document = self.transforms()
        document["sequences"]["000237"][0]["image_name"] = "image_1.png"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "match inference image"):
            self.world_diagnostic(path)
        self.assertFalse(self.output.exists())

    def test_world_mode_rejects_reflection_and_nonfinite_translation(self):
        self.sequence()
        path, document = self.transforms()
        document["sequences"]["000237"][0]["rotation"] = [[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "proper orthonormal"):
            self.world_diagnostic(path)
        document["sequences"]["000237"][0]["rotation"] = np.eye(3).tolist()
        document["sequences"]["000237"][0]["translation"] = [float("inf"), 0., 0.]
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.world_diagnostic(path)

    def test_world_policy_cannot_be_silently_ignored_or_omitted(self):
        self.sequence()
        with self.assertRaisesRegex(ValueError, "requires --world-transforms"):
            self.world_diagnostic(None)
        path, _ = self.transforms()
        with self.assertRaisesRegex(ValueError, "requires legacy_native_xy_to_world"):
            exporter.export_submission(
                self.inputs, self.output, "legacy_native_xy", expected_ids=None,
                diagnostic_single_id="000237", world_transforms=path)

    def test_world_transform_manifest_rejects_missing_frames_and_unverified_source(self):
        self.sequence()
        path, document = self.transforms()
        document["source_verification"] = ""
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_verification"):
            self.world_diagnostic(path)
        document["source_verification"] = "Training provenance fixture"
        document["sequences"]["000237"].pop()
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "30 frames"):
            self.world_diagnostic(path)

    def test_official_world_export_keeps_both_archives_to_ten_txt_files(self):
        for video_id in exporter.OFFICIAL_IDS:
            self.sequence(video_id)
        path, _ = self.transforms(exporter.OFFICIAL_IDS)
        result = exporter.export_submission(
            self.inputs, self.output, "legacy_native_xy_to_world", "official",
            world_transforms=path)
        for key in ("zip", "native_backup_zip"):
            with zipfile.ZipFile(result[key]) as archive:
                self.assertEqual(archive.namelist(), [
                    "result/{}.txt".format(video_id) for video_id in exporter.OFFICIAL_IDS])


if __name__ == "__main__":
    unittest.main()
