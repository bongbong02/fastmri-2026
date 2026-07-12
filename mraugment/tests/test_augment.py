import argparse
import unittest

import torch
import numpy as np

from mraugment import MRAugment, add_mraugment_args
from utils.data.transforms import DataTransform


def make_args(*extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    add_mraugment_args(parser)
    return parser.parse_args(list(extra))


class MRAugmentTest(unittest.TestCase):
    def test_disabled_is_identity(self):
        aug = MRAugment(make_args())
        kspace = torch.randn(3, 12, 10, dtype=torch.complex64)
        output, target = aug(kspace, (8, 8))
        self.assertIs(output, kspace)
        self.assertIsNone(target)

    def test_augmented_pair_has_expected_shape_and_is_finite(self):
        aug = MRAugment(make_args("--mraugment", "--aug-schedule", "constant", "--aug-strength", "1"))
        kspace = torch.randn(4, 16, 12, dtype=torch.complex64)
        output, target = aug(kspace, (10, 10))
        self.assertEqual(output.shape, kspace.shape)
        self.assertEqual(target.shape, (10, 10))
        self.assertTrue(torch.isfinite(torch.view_as_real(output)).all())
        self.assertTrue(torch.isfinite(target).all())

    def test_seed_reproducibility(self):
        args = make_args("--mraugment", "--aug-schedule", "constant", "--aug-strength", "1")
        kspace = torch.randn(2, 14, 14, dtype=torch.complex64)
        a, _ = MRAugment(args)(kspace, (12, 12))
        b, _ = MRAugment(args)(kspace, (12, 12))
        torch.testing.assert_close(a, b)

    def test_ramp_reaches_requested_strength(self):
        aug = MRAugment(make_args("--mraugment", "--aug-schedule", "ramp", "--aug-strength", "0.6"))
        self.assertAlmostEqual(aug.probability(), 0.2)
        aug.set_epoch(2)
        self.assertAlmostEqual(aug.probability(), 0.6)

    def test_training_transform_masks_augmented_full_kspace(self):
        args = make_args("--mraugment", "--aug-schedule", "constant", "--aug-strength", "1")
        transform = DataTransform(False, "max", MRAugment(args))
        mask = np.zeros(12, dtype=np.float32)
        mask[4:8] = 1
        kspace = (np.random.randn(2, 16, 12) + 1j * np.random.randn(2, 16, 12)).astype(np.complex64)
        target = np.zeros((10, 10), dtype=np.float32)
        out_mask, masked, out_target, maximum, *_ = transform(mask, kspace, target, {"max": 2.0}, "x.h5", 0)
        self.assertEqual(masked.shape, (2, 16, 12, 2))
        self.assertEqual(out_target.shape, (10, 10))
        self.assertEqual(out_mask.shape, (1, 1, 12, 1))
        self.assertEqual(maximum, 2.0)
        self.assertTrue(torch.count_nonzero(masked[:, :, :4]) == 0)
        self.assertTrue(torch.count_nonzero(masked[:, :, 8:]) == 0)


if __name__ == "__main__":
    unittest.main()
