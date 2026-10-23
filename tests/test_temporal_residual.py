"""CPU regression of the real decoder.forward with a controlled temporal unit."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


def decoder_class():
    """Extract the actual class while avoiding constructor CUDA side effects."""
    path = Path(__file__).resolve().parents[1] / "lib/models/mono3dvg/mono3dvg_transformer.py"
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "Mono3DVGDecoder"]
    if len(nodes) != 1:
        raise AssertionError("Expected one actual decoder definition")
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Mono3DVGDecoder"]


@unittest.skipIf(torch is None, "PyTorch unavailable; run with training Python on CPU")
class TemporalResidualTests(unittest.TestCase):
    def decoder(self, flag=None, bad_shape=False):
        class SpatialLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.))

            def forward(self, value, *args):
                return self.scale * value

        class TemporalUnit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value, memory):
                self.calls += 1
                temporal = value.squeeze(2) * 0.
                if bad_shape:
                    temporal = temporal.expand(1, 2, 4)
                memory = torch.zeros(1, 2, 4) if memory is None else memory
                return temporal, memory + 1.

        cls = decoder_class()
        decoder = cls.__new__(cls)
        torch.nn.Module.__init__(decoder)
        decoder.layers = torch.nn.ModuleList([SpatialLayer()])
        decoder.dim_embed = torch.nn.ModuleList([torch.nn.Linear(4, 3)])
        decoder.bbox_embed = None
        decoder.return_intermediate = True
        decoder.memory_tokens = None
        decoder.TTM_model = TemporalUnit()
        if flag is not None:
            decoder.official_temporal_residual = flag
        return decoder

    def forward(self, decoder, value):
        # Only bypass this historical .cuda() transfer. The real forward math,
        # residual branch, shape check and TTM memory assignment all execute.
        with patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor):
            return decoder(value, torch.full((1, 1, 2), .5), torch.zeros(1, 1, 4),
                           torch.tensor([[1, 1]]), torch.tensor([0]), torch.ones(1, 1, 2))[0]

    def test_residual_retains_input_signal_and_gradient_when_temporal_path_collapses(self):
        decoder = self.decoder(True)
        value = torch.tensor([[[1., 2., 3., 4.]]], requires_grad=True)
        result = self.forward(decoder, value)
        torch.testing.assert_close(result, (value * 2).unsqueeze(0), rtol=0, atol=0)
        result.sum().backward()
        torch.testing.assert_close(value.grad, torch.full_like(value, 2.), rtol=0, atol=0)
        self.assertGreater(float(decoder.layers[0].scale.grad.abs()), 0.)

    def test_default_and_false_preserve_inherited_temporal_only_behavior(self):
        value = torch.arange(4.).reshape(1, 1, 4).requires_grad_()
        default, explicit_false = self.decoder(), self.decoder(False)
        a, b = self.forward(default, value), self.forward(explicit_false, value)
        torch.testing.assert_close(a, torch.zeros_like(a), rtol=0, atol=0)
        torch.testing.assert_close(b, a, rtol=0, atol=0)
        a.sum().backward()
        self.assertEqual(float(value.grad.abs().max()), 0.)

    def test_temporal_memory_still_updates_and_state_keys_do_not_change(self):
        decoder = self.decoder(False)
        keys_before = set(decoder.state_dict())
        value = torch.ones(1, 1, 4)
        self.forward(decoder, value)
        self.assertEqual(decoder.TTM_model.calls, 1)
        self.assertEqual(float(decoder.memory_tokens.mean()), 1.)
        decoder.official_temporal_residual = True
        self.forward(decoder, value)
        self.assertEqual(decoder.TTM_model.calls, 2)
        self.assertEqual(float(decoder.memory_tokens.mean()), 2.)
        self.assertEqual(set(decoder.state_dict()), keys_before)

    def test_residual_refuses_silent_broadcast_of_temporal_query_axis(self):
        decoder = self.decoder(True, bad_shape=True)
        with self.assertRaisesRegex(ValueError, "identical spatial/temporal shapes"):
            self.forward(decoder, torch.ones(1, 1, 4))


if __name__ == "__main__":
    unittest.main()
