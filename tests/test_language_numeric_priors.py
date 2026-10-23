"""Pure CPU tests; no dataset, model, torch, or numpy is imported."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.language_numeric_priors import (apply_numeric_priors, parse_initial_priors,
                                           parse_initial_numeric_priors)


class LanguageNumericPriorsTests(unittest.TestCase):
    @staticmethod
    def decoded(xyz, dims):
        return {"label_position_native_xyz": list(xyz), "dimensions_lwh": list(dims),
                "values_lwhxyz": list(dims) + list(xyz),
                "camera_center_xyz": [1.0, 2.0, 30.0], "camera_z": 30.0,
                "normalized_center_xy": [0.25, 0.75], "other": {"flag": True}}

    def test_semantic_lwh_and_initial_only_spans(self):
        text = ("A van measuring 0.9 meters in height, 4.5 meters in length, and "
                "1.7 meters in width is initially parked at a distance of 20.3 meters, "
                "at an azimuth of -175 degrees. As the video progresses it is "
                "30.0 meters away, 1.8 meters tall, 5 meters long, and 2 meters wide.")
        p = parse_initial_priors(text)
        self.assertEqual(p["status"], "valid")
        self.assertEqual(p["range_m"], 20.3)
        self.assertEqual(p["dimensions_lwh"], [4.5, 1.7, 0.9])
        self.assertNotIn("azimuth", p["fields"])
        for field in p["fields"].values():
            for c in field["candidates"]:
                self.assertEqual(text[slice(*c["span"])], c["snippet"])
        self.assertLess(p["initial_span"][1], text.index("30.0 meters"))

    def test_away_begins_and_dimension_word_order(self):
        p = parse_initial_priors(
            "A bus, 2 meters tall, 7.3 meters long, 1.9 meters wide, begins its "
            "journey 20.5 meters away from me. Initially, it is moving. "
            "By the end it is 50 meters from me.")
        self.assertEqual(p["range_m"], 20.5)
        self.assertEqual(p["dimensions_lwh"], [7.3, 1.9, 2.0])
        p = parse_initial_priors("Initially a car has width of 1.8 metres, "
                                 "height of 1.5 metres, length of 4 metres, "
                                 "and is 12 metres from me.")
        self.assertEqual(p["range_m"], 12)
        self.assertEqual(p["dimensions_lwh"], [4, 1.8, 1.5])

    def test_conflicts_fail_per_field_and_consistent_repeats_are_auditable(self):
        p = parse_initial_priors("Initially, at a distance of 10 meters and "
                                 "20 meters away, it is 1 meters tall, "
                                 "2 meters wide, 4 meters long, and 5 meters long.")
        self.assertEqual(p["status"], "ambiguous")
        self.assertIsNone(p["range_m"])
        self.assertIsNone(p["dimensions_lwh"])
        self.assertEqual(p["ambiguity"], {"range": True, "dimensions": True})
        p = parse_initial_priors("Initially at a distance of 10 meters, "
                                 "still 10.0 meters away.")
        self.assertEqual(p["range_m"], 10)
        self.assertEqual(len(p["fields"]["range"]["candidates"]), 2)

    def test_missing_initial_anchor_or_final_only_cannot_supply_range(self):
        for text in ("The car is 12 meters away.",
                     "At the end it is 12 meters away. Initially it was moving.",
                     "Initially it is moving.\n\nThe car is 12 meters away.",
                     "Initially it is moving. Later it is 12 meters away."):
            self.assertIsNone(parse_initial_priors(text)["range_m"])
        self.assertEqual(parse_initial_priors(None)["status"], "invalid_input")
        self.assertIs(parse_initial_numeric_priors, parse_initial_priors)

    def test_invalid_numeric_candidates_never_apply(self):
        for token in ("0", "-2", "1e309"):
            p = parse_initial_priors("Initially at a distance of %s meters, "
                                     "with %s meters tall, 4 meters long, "
                                     "2 meters wide." % (token, token))
            self.assertIsNone(p["range_m"])
            self.assertIsNone(p["dimensions_lwh"])
            self.assertEqual(p["fields"]["range"]["status"], "invalid")
            json.dumps(p, allow_nan=False)
        p = parse_initial_priors("Initially 1,000 meters away and 1.2.3 meters tall.")
        self.assertIsNone(p["range_m"])
        self.assertIsNone(p["dimensions_lwh"])

    def test_intervals_negations_and_inequalities_are_not_exact_priors(self):
        for phrase in ("between 10 and 20 meters away", "from 10 to 20 meters away",
                       "not 20 meters away", "at least 20 meters away",
                       "10 or 20 meters away", "approximately 20 meters away",
                       "approximately at a distance of 20 meters",
                       "10\u201320 meters away", "10 - 20 meters away"):
            p = parse_initial_priors("Initially " + phrase + ".")
            self.assertIsNone(p["range_m"], phrase)
            self.assertEqual(p["fields"]["range"]["status"], "ambiguous", phrase)
        for length in ("not 4 meters long", "between 3 and 4 meters long",
                       "3\u20134 meters long"):
            p = parse_initial_priors("Initially 20 meters away, 1 meter tall, "
                                     "2 meters wide, " + length + ".")
            self.assertEqual(p["range_m"], 20)
            self.assertIsNone(p["dimensions_lwh"], length)
            self.assertTrue(p["ambiguity"]["dimensions"], length)
        p = parse_initial_priors("Initially moving. Now it is 20 meters away.")
        self.assertIsNone(p["range_m"])

    def test_rescale_preserves_native_direction_and_copies_all_inputs(self):
        xyz, dims = [-3.0, 4.0, 0.0], [4.1, 1.8, 1.5]
        p = parse_initial_priors("Initially 10 meters from me, 1.2 meters tall, "
                                 "4.2 meters long and 1.6 meters wide.")
        decoded = self.decoded(xyz, dims)
        before = copy.deepcopy((xyz, dims, p, decoded))
        result = apply_numeric_priors(decoded, p, use_dimensions=False)
        self.assertEqual(result["label_position_native_xyz"], [-6.0, 8.0, 0.0])
        self.assertAlmostEqual(math.hypot(*result["label_position_native_xyz"]), 10)
        self.assertTrue(result["numeric_priors"]["range_applied"])
        self.assertFalse(result["numeric_priors"]["dimensions_applied"])
        self.assertEqual(result["dimensions_lwh"], dims)
        self.assertEqual(result["values_lwhxyz"], dims + [-6, 8, 0])
        self.assertEqual((xyz, dims, p, decoded), before)
        result["dimensions_lwh"][0] = 100
        result["other"]["flag"] = False
        self.assertTrue(decoded["other"]["flag"])
        self.assertEqual(dims[0], 4.1)

    def test_optional_dimensions_and_independent_field_fallbacks(self):
        p = parse_initial_priors("Initially at a distance of 10 meters, "
                                 "1.2 meters tall, 4.2 meters long, 1.6 meters wide.")
        result = apply_numeric_priors(self.decoded([0, 0, 0], [1, 1, 1]), p)
        self.assertFalse(result["numeric_priors"]["range_applied"])
        self.assertEqual(result["label_position_native_xyz"], [0, 0, 0])
        self.assertTrue(result["numeric_priors"]["dimensions_applied"])
        self.assertEqual(result["dimensions_lwh"], [4.2, 1.6, 1.2])
        self.assertEqual(result["values_lwhxyz"], [4.2, 1.6, 1.2, 0, 0, 0])
        self.assertIn("zero", result["numeric_priors"]["reasons"][0])
        result = apply_numeric_priors(self.decoded([1, 2, 3], [1, 1, float("inf")]), p,
                                      use_range=False, use_dimensions=True)
        self.assertFalse(result["numeric_priors"]["range_applied"])
        self.assertFalse(result["numeric_priors"]["dimensions_applied"])
        self.assertTrue(math.isinf(result["dimensions_lwh"][2]))

    def test_bad_vectors_and_unparsed_prior_fallback_unchanged(self):
        p = parse_initial_priors("Initially 20 meters away.")
        for xyz in ([0, float("nan"), 1], [1, float("inf"), 2],
                    [1, "3", 4], [True, 1, 2]):
            result = apply_numeric_priors(self.decoded(xyz, [1, 1, 1]), p)
            self.assertFalse(result["numeric_priors"]["range_applied"])
            self.assertTrue(result["numeric_priors"]["reasons"])
            for before, after in zip(xyz, result["label_position_native_xyz"]):
                if isinstance(before, float) and math.isnan(before):
                    self.assertTrue(math.isnan(after))
                else:
                    self.assertEqual(before, after)
        for prior in (None, {}, {"range_m": 50}, parse_initial_priors("Initially moving.")):
            result = apply_numeric_priors(self.decoded((1, 2, 3), (4, 2, 1)), prior)
            self.assertFalse(result["numeric_priors"]["range_applied"])
            self.assertEqual(result["label_position_native_xyz"], [1, 2, 3])

    def test_native_rescaling_removes_stale_camera_geometry(self):
        decoded = self.decoded([3, 4, 0], [4, 2, 1])
        result = apply_numeric_priors(decoded, parse_initial_priors("Initially 20 meters away."))
        self.assertEqual(result["label_position_native_xyz"], [12, 16, 0])
        self.assertFalse(result["camera_metrics_available"])
        for key in ("camera_center_xyz", "camera_z", "normalized_center_xy"):
            self.assertNotIn(key, result)
            self.assertIn(key, decoded)

    def test_dims_only_retains_available_camera_geometry(self):
        decoded = self.decoded([3, 4, 0], [4, 2, 1])
        p = parse_initial_priors("Initially 20 meters away, 1.2 meters tall, "
                                 "4.2 meters long, and 1.6 meters wide.")
        result = apply_numeric_priors(decoded, p, use_range=False)
        self.assertTrue(result["camera_metrics_available"])
        self.assertEqual(result["values_lwhxyz"], [4.2, 1.6, 1.2, 3, 4, 0])
        for key in ("camera_center_xyz", "camera_z", "normalized_center_xy"):
            self.assertEqual(result[key], decoded[key])


if __name__ == "__main__":
    unittest.main()
