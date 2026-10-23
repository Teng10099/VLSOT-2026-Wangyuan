import unittest
try:
    import torch
except ModuleNotFoundError:
    torch = None
if torch is not None:
    from lib.losses.official_geometry import OfficialGeometryHead, OfficialGeometryCriterion


@unittest.skipIf(torch is None, "PyTorch runtime required")
class GeometrySupervisionTest(unittest.TestCase):
    def test_projection_outside_image_remains_trainable(self):
        head = OfficialGeometryHead(8)
        with torch.no_grad():
            head.layers[-1].weight.zero_()
            head.layers[-1].bias[:2] = torch.tensor([1.1, -0.1])
        output = head(torch.ones(1, 1, 8))
        self.assertTrue(torch.allclose(output['pred_boxes'][0, 0, :2], torch.tensor([1.1, -0.1])))
        self.assertGreater(float(output['pred_depth'][0, 0, 0]), 0)
        self.assertTrue(bool((output['pred_3d_dim'] > 0).all()))

    def test_only_available_labels_and_finite_gradient(self):
        head = OfficialGeometryHead(8)
        loss_fn = OfficialGeometryCriterion()
        features = torch.randn(1, 1, 8, requires_grad=True)
        targets = [{'projected_center': torch.tensor([[1.1, -.05]]),
                    'depth': torch.tensor([[150.]]),
                    'size_3d': torch.tensor([[1.5, 1.7, 4.3]])}]
        losses = loss_fn(head(features), targets)
        total = sum(losses[k] * w for k, w in loss_fn.weight_dict.items())
        total.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertNotIn('loss_bbox', losses)
        self.assertNotIn('loss_angle', losses)
        self.assertNotIn('loss_depth_map', losses)

    def test_can_learn_metric_depth_beyond_legacy_sixty_meter_limit(self):
        torch.manual_seed(444)
        head = OfficialGeometryHead(8)
        loss_fn = OfficialGeometryCriterion()
        optim = torch.optim.Adam(head.parameters(), lr=.03)
        features = torch.ones(1, 1, 8)
        targets = [{'projected_center': torch.tensor([[.35, .25]]),
                    'depth': torch.tensor([[140.]]),
                    'size_3d': torch.tensor([[1.3, 1.7, 4.3]])}]
        values = []
        for _ in range(80):
            optim.zero_grad()
            losses = loss_fn(head(features), targets)
            total = sum(losses[k] * w for k, w in loss_fn.weight_dict.items())
            values.append(float(total.detach()))
            total.backward()
            optim.step()
        self.assertLess(sum(values[-5:]) / 5, values[0] * .15)
        self.assertGreater(float(head(features)['pred_depth'][0, 0, 0]), 100.)


if __name__ == '__main__':
    unittest.main()
