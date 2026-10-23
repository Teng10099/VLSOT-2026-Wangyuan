"""Run the actual differential-attention class on CPU, without CUDA imports.

Only its source definitions and helpers are compiled from the checked-in model;
the attention math is not reimplemented by these tests.
"""
import ast
import math
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


def actual_attention_class():
    path = Path(__file__).resolve().parents[1] / "lib/models/mono3dvg/mono3dvg_transformer.py"
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"RMSNorm", "repeat_kv", "lambda_init_fn", "MultiheadDiffAttn"}
    definitions = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in definitions} != names:
        raise AssertionError("Actual attention definitions changed; review isolated CPU import")
    module = ast.Module(body=definitions, type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "math": math}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["MultiheadDiffAttn"]


@unittest.skipIf(torch is None, "PyTorch unavailable; run with training Python and CUDA_VISIBLE_DEVICES='' ")
class PaddingMaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(417)
        self.model = actual_attention_class()(16, 2, 1).eval()
        self.q, self.k, self.v = torch.randn(2, 1, 16), torch.randn(4, 1, 16), torch.randn(4, 1, 16)
        self.mask = torch.tensor([[False, False, True, True]])

    def test_padding_token_values_cannot_change_corrected_output(self):
        self.model.correct_padding_mask = True
        expected = self.model(self.q, self.k, self.v, self.mask)
        changed_k, changed_v = self.k.clone(), self.v.clone()
        changed_k[2:] = torch.randn_like(changed_k[2:]) * 1000
        changed_v[2:] = torch.randn_like(changed_v[2:]) * 1000
        actual = self.model(self.q, changed_k, changed_v, self.mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.model.correct_padding_mask = False
        old = self.model(self.q, self.k, self.v, self.mask)
        changed_old = self.model(self.q, changed_k, changed_v, self.mask)
        self.assertGreater(float((old - changed_old).abs().max()), 1e-5)

    def test_padded_keys_values_get_zero_gradient_but_valid_tokens_learn(self):
        self.model.correct_padding_mask = True
        k, v = self.k.clone().requires_grad_(), self.v.clone().requires_grad_()
        result = self.model(self.q, k, v, self.mask)
        result[..., 0].sum().backward()
        self.assertTrue(torch.isfinite(k.grad).all())
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertEqual(float(k.grad[2:].abs().max()), 0.)
        self.assertEqual(float(v.grad[2:].abs().max()), 0.)
        self.assertGreater(float(v.grad[:2].abs().max()), 0.)

    def test_mask_free_additive_causal_behavior_and_weight_keys_remain_compatible(self):
        before = set(self.model.state_dict())
        self.model.correct_padding_mask = False
        old = self.model(self.q, self.k, self.v, None)
        self.model.correct_padding_mask = True
        corrected = self.model(self.q, self.k, self.v, None)
        torch.testing.assert_close(corrected, old, rtol=0, atol=0)
        self.assertEqual(set(self.model.state_dict()), before)

    def test_wrong_boolean_mask_shape_fails_loudly(self):
        self.model.correct_padding_mask = True
        with self.assertRaisesRegex(ValueError, "source_length"):
            self.model(self.q, self.k, self.v, torch.tensor([[True, False]]))


if __name__ == "__main__":
    unittest.main()
