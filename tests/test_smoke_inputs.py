"""Check that inference consumes paths/text rather than numeric target labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "smoke_infer", Path(__file__).resolve().parents[1] / "tools/smoke_infer.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeInputTests(unittest.TestCase):
    def test_target_labels_are_not_required_or_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000237.json"
            record = {"videoID": 237, "description": "A WHITE CAR"}
            for i in range(30):
                # Only the paths are present: no boxes, dimensions, pose or class.
                record["frame{}".format(i)] = [True, r"labels\a.json", r"images\a.jpg"]
            path.write_text(json.dumps([record]), encoding="utf-8")
            video_id, text, frames = smoke.read_sequence(path)
            self.assertEqual(video_id, "000237")
            self.assertEqual(text, "a white car")
            self.assertEqual(len(frames), 30)
            self.assertEqual(set(frames[0]), {"frame_index", "image_name", "calibration_name"})
            self.assertEqual(frames[0]["calibration_name"], "a.json")

    def test_incomplete_clip_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000237.json"
            path.write_text(json.dumps({"description": "car", "frame0": [True, "a", "b"]}))
            with self.assertRaisesRegex(ValueError, "Missing frame"):
                smoke.read_sequence(path)

    def test_language_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000237.json"
            path.write_text(json.dumps({"description": {"style": "car"}}))
            with self.assertRaisesRegex(ValueError, "description"):
                smoke.read_sequence(path)


if __name__ == "__main__":
    unittest.main()
