"""Check which feature paths may learn without importing torch or CUDA."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_official_selection", str(ROOT / "train_official.py"))
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class Parameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value


class Model:
    def __init__(self):
        self.parameters = {name: Parameter() for name in (
            "official_geometry_head.layers.0.weight", "official_geometry_head.layers.2.bias",
            "backbone.backbone.model.layers.0.weight", "text_encoder.encoder.layer.0.weight",
            "depth_predictor.depth_encoder.weight", "input_proj.0.0.weight", "resizer.fc.weight",
            "mono3dvg_transformer.TextGuidedAdapter.orig_proj.layers.0.weight",
            "mono3dvg_transformer.encoder.layers.0.weight",
            "mono3dvg_transformer.decoder.layers.0.weight",
            "mono3dvg_transformer.decoder.TTM_model.pre1.conv.weight",
            "class_embed.0.weight",
        )}

    def named_parameters(self):
        return self.parameters.items()


class TrainableSelectionTests(unittest.TestCase):
    def test_fusion_can_adapt_visual_and_text_projections_but_freezes_large_encoders(self):
        model = Model()
        names = set(entry.configure_trainable(model, "fusion"))
        self.assertIn("input_proj.0.0.weight", names)
        self.assertIn("resizer.fc.weight", names)
        self.assertIn("mono3dvg_transformer.TextGuidedAdapter.orig_proj.layers.0.weight", names)
        self.assertIn("mono3dvg_transformer.encoder.layers.0.weight", names)
        self.assertIn("mono3dvg_transformer.decoder.TTM_model.pre1.conv.weight", names)
        self.assertNotIn("backbone.backbone.model.layers.0.weight", names)
        self.assertNotIn("text_encoder.encoder.layer.0.weight", names)
        self.assertNotIn("depth_predictor.depth_encoder.weight", names)
        self.assertNotIn("class_embed.0.weight", names)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in names)

    def test_head_only_and_decoder_warmups_remain_distinct(self):
        model = Model()
        head = set(entry.configure_trainable(model, "head"))
        self.assertEqual(len(head), 2)
        self.assertTrue(all(name.startswith("official_geometry_head.") for name in head))
        decoder = set(entry.configure_trainable(model, "decoder"))
        self.assertTrue(head < decoder)
        self.assertNotIn("input_proj.0.0.weight", decoder)
        self.assertNotIn("resizer.fc.weight", decoder)
        self.assertIn("mono3dvg_transformer.decoder.layers.0.weight", decoder)

    def test_parser_keeps_separate_head_lr_residual_and_initialization_arguments(self):
        args = entry.parse_args([
            "--config", "cfg", "--source-project", "src", "--checkpoint", "weights",
            "--data-root", "data", "--calibration-root", "calib", "--output", "run",
            "--label-center-reference", "geometric_center", "--trainable", "fusion",
            "--lr", ".00002", "--head-lr", ".0005", "--head-init-std", ".02", "--temporal-residual",
        ])
        self.assertEqual(args.trainable, "fusion")
        self.assertEqual(args.lr, .00002)
        self.assertEqual(args.head_lr, .0005)
        self.assertEqual(args.head_init_std, .02)
        self.assertTrue(args.temporal_residual)


if __name__ == "__main__":
    unittest.main()
