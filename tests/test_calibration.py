"""
Unit tests for probability calibration metrics and reliability diagram data.

Tests:
1. Perfect predictions: Brier=0, NLL~0, ECE=0.
2. Completely incorrect predictions: Brier=1 (binary) / 2 (multiclass), high NLL, ECE=1.
3. Well-calibrated synthetic predictions: ECE=0 across defined bins.
4. Numerical distinction between top-label ECE and positive-class ECE.
5. Empty bin handling and edge cases without ZeroDivisionError.
6. Mathematical relationship: BS_multi = 2 * BS_binary for 2-class normalized distributions.
7. Input validation: invalid labels, out-of-range probabilities, shape mismatches,
   unnormalized rows, non-finite values, and empty arrays.
8. Support for PyTorch tensors, NumPy arrays, and Python lists.
"""

import math
import unittest
import numpy as np
import torch

from src.evaluation.calibration import (
    compute_binary_brier_score,
    compute_multiclass_brier_score,
    compute_brier_score,
    compute_nll,
    compute_reliability_diagram_data,
    compute_ece,
    compute_positive_class_ece,
    compute_top_label_ece,
    compute_calibration_metrics,
)


class TestCalibrationMetrics(unittest.TestCase):
    """Test suite verifying calibration metrics against analytical ground truths."""

    def test_perfect_predictions(self):
        """When probabilities match ground truth exactly, calibration error should be 0."""
        y_true = [0, 1, 0, 1]
        probs_1d = [0.0, 1.0, 0.0, 1.0]
        probs_2d = [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        # Binary and multiclass Brier score
        bs_bin = compute_binary_brier_score(y_true, probs_1d)
        bs_multi = compute_multiclass_brier_score(y_true, probs_2d)
        self.assertAlmostEqual(bs_bin, 0.0, places=6)
        self.assertAlmostEqual(bs_multi, 0.0, places=6)

        # NLL should be close to 0 (bounded by eps=1e-15)
        nll = compute_nll(y_true, probs_2d)
        self.assertLess(nll, 1e-10)

        # ECE for both strategies should be 0
        ece_pos = compute_positive_class_ece(y_true, probs_1d, num_bins=10)
        ece_top = compute_top_label_ece(y_true, probs_2d, num_bins=10)
        self.assertAlmostEqual(ece_pos, 0.0, places=6)
        self.assertAlmostEqual(ece_top, 0.0, places=6)

        # Reliability diagram data
        rel_pos = compute_reliability_diagram_data(y_true, probs_1d, num_bins=10, strategy="positive_class")
        self.assertEqual(rel_pos["sample_counts"][0], 2)  # bin [0.0, 0.1) has 2 samples
        self.assertEqual(rel_pos["sample_counts"][9], 2)  # bin [0.9, 1.0] has 2 samples
        self.assertAlmostEqual(rel_pos["bin_confidences"][0], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["bin_accuracies"][0], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["bin_confidences"][9], 1.0, places=6)
        self.assertAlmostEqual(rel_pos["bin_accuracies"][9], 1.0, places=6)
        self.assertAlmostEqual(rel_pos["ece"], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["mce"], 0.0, places=6)

    def test_completely_incorrect_predictions(self):
        """When probabilities are completely wrong with 100% confidence."""
        y_true = [0, 1]
        probs_2d = [
            [0.0, 1.0],  # true 0, predicted 1 with p=1.0
            [1.0, 0.0],  # true 1, predicted 0 with p=1.0
        ]
        probs_1d = [1.0, 0.0]

        bs_bin = compute_binary_brier_score(y_true, probs_1d)
        bs_multi = compute_multiclass_brier_score(y_true, probs_2d)
        self.assertAlmostEqual(bs_bin, 1.0, places=6)
        self.assertAlmostEqual(bs_multi, 2.0, places=6)

        # NLL should be very large: -log(1e-15) ≈ 34.54
        nll = compute_nll(y_true, probs_2d, eps=1e-15)
        expected_nll = -math.log(1e-15)
        self.assertAlmostEqual(nll, expected_nll, places=4)

        # ECE should be 1.0
        ece_pos = compute_positive_class_ece(y_true, probs_1d, num_bins=10)
        ece_top = compute_top_label_ece(y_true, probs_2d, num_bins=10)
        self.assertAlmostEqual(ece_pos, 1.0, places=6)
        self.assertAlmostEqual(ece_top, 1.0, places=6)

    def test_well_calibrated_synthetic_distribution(self):
        """
        Construct a synthetic dataset where in each bin, the true positive rate
        matches the predicted probability exactly:
        - 100 samples with P(FAKE) = 0.2: 20 positive (1), 80 negative (0)
        - 100 samples with P(FAKE) = 0.5: 50 positive (1), 50 negative (0)
        - 100 samples with P(FAKE) = 0.8: 80 positive (1), 20 negative (0)
        """
        y_group1 = [1] * 20 + [0] * 80
        p_group1 = [0.2] * 100

        y_group2 = [1] * 50 + [0] * 50
        p_group2 = [0.5] * 100

        y_group3 = [1] * 80 + [0] * 20
        p_group3 = [0.8] * 100

        y_true = np.array(y_group1 + y_group2 + y_group3)
        y_probs = np.array(p_group1 + p_group2 + p_group3)

        # Positive-class ECE should be 0.0
        ece_pos = compute_positive_class_ece(y_true, y_probs, num_bins=10)
        self.assertAlmostEqual(ece_pos, 0.0, places=6)

        # Top-label ECE should also be 0.0:
        # Group 1: max conf is 0.8 (REAL), accuracy is 80/100 = 0.8
        # Group 2: max conf is 0.5 (either), accuracy is 50/100 = 0.5
        # Group 3: max conf is 0.8 (FAKE), accuracy is 80/100 = 0.8
        ece_top = compute_top_label_ece(y_true, y_probs, num_bins=10)
        self.assertAlmostEqual(ece_top, 0.0, places=6)

        # Check reliability diagram data
        rel_pos = compute_reliability_diagram_data(y_true, y_probs, num_bins=10, strategy="positive_class")
        self.assertAlmostEqual(rel_pos["ece"], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["mce"], 0.0, places=6)
        # Bins 2, 5, 8 (0.2, 0.5, 0.8) should each have 100 samples and gap 0.0
        self.assertEqual(rel_pos["sample_counts"][2], 100)
        self.assertEqual(rel_pos["sample_counts"][5], 100)
        self.assertEqual(rel_pos["sample_counts"][8], 100)
        self.assertAlmostEqual(rel_pos["bin_gaps"][2], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["bin_gaps"][5], 0.0, places=6)
        self.assertAlmostEqual(rel_pos["bin_gaps"][8], 0.0, places=6)

    def test_distinction_between_top_label_and_positive_class_ece(self):
        """
        Verify that Top-Label ECE and Positive-Class ECE are distinct metrics
        and return different numerical values on asymmetric distributions.

        Setup:
        - Sample 1: y=0, P(FAKE)=0.2 -> P(REAL)=0.8 (predicted REAL, correct, conf=0.8)
        - Sample 2: y=1, P(FAKE)=0.2 -> P(REAL)=0.8 (predicted REAL, wrong, conf=0.8)
        - Sample 3: y=1, P(FAKE)=0.8 -> P(REAL)=0.2 (predicted FAKE, correct, conf=0.8)

        Top-label perspective:
        All 3 samples have top confidence 0.8.
        Samples 1 and 3 are correct (accuracy = 2/3 ≈ 0.6667).
        Gap = |2/3 - 0.8| = 0.13333.
        ECE_top = 0.13333.

        Positive-class perspective:
        - Bin [0.2]: Samples 1 & 2 have P(FAKE)=0.2. One is FAKE, so accuracy = 1/2 = 0.5.
          Gap = |0.5 - 0.2| = 0.3. Weight = 2/3.
        - Bin [0.8]: Sample 3 has P(FAKE)=0.8. It is FAKE, so accuracy = 1/1 = 1.0.
          Gap = |1.0 - 0.8| = 0.2. Weight = 1/3.
        ECE_pos = (2/3)*0.3 + (1/3)*0.2 = 0.2 + 0.06667 = 0.26667.

        Since 0.13333 != 0.26667, the two metrics are demonstrably non-interchangeable.
        """
        y_true = np.array([0, 1, 1])
        y_probs = np.array([
            [0.8, 0.2],
            [0.8, 0.2],
            [0.2, 0.8],
        ])

        ece_top = compute_top_label_ece(y_true, y_probs, num_bins=10)
        ece_pos = compute_positive_class_ece(y_true, y_probs, num_bins=10)

        expected_ece_top = abs((2.0 / 3.0) - 0.8)
        expected_ece_pos = (2.0 / 3.0) * abs(0.5 - 0.2) + (1.0 / 3.0) * abs(1.0 - 0.8)

        self.assertAlmostEqual(ece_top, expected_ece_top, places=5)
        self.assertAlmostEqual(ece_pos, expected_ece_pos, places=5)
        self.assertNotAlmostEqual(ece_top, ece_pos, places=3)
        self.assertAlmostEqual(abs(ece_pos - ece_top), 0.133333, places=4)

    def test_empty_bins_handling(self):
        """Verify that empty bins do not produce NaN or ZeroDivisionError in ECE/MCE."""
        y_true = np.array([0, 1])
        y_probs = np.array([0.15, 0.85])  # Only populates bins 1 and 8

        rel = compute_reliability_diagram_data(y_true, y_probs, num_bins=10, strategy="positive_class")

        # Non-occupied bins should have count 0 and NaN values
        self.assertEqual(rel["sample_counts"][0], 0)
        self.assertTrue(np.isnan(rel["bin_confidences"][0]))
        self.assertTrue(np.isnan(rel["bin_accuracies"][0]))
        self.assertTrue(np.isnan(rel["bin_gaps"][0]))

        # Occupied bins
        self.assertEqual(rel["sample_counts"][1], 1)
        self.assertEqual(rel["sample_counts"][8], 1)
        self.assertFalse(np.isnan(rel["bin_confidences"][1]))
        self.assertFalse(np.isnan(rel["bin_confidences"][8]))

        # Overall ECE and MCE must be finite floats
        self.assertTrue(np.isfinite(rel["ece"]))
        self.assertTrue(np.isfinite(rel["mce"]))

    def test_brier_score_binary_vs_multiclass_relation(self):
        """
        For binary classification with probabilities summing to 1:
        BS_multi = 2 * BS_binary.
        """
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100)
        p_fake = rng.uniform(0.0, 1.0, size=100)
        probs_2d = np.column_stack([1.0 - p_fake, p_fake])

        bs_bin = compute_binary_brier_score(y_true, p_fake)
        bs_multi = compute_multiclass_brier_score(y_true, probs_2d)

        self.assertAlmostEqual(bs_multi, 2.0 * bs_bin, places=6)
        self.assertAlmostEqual(compute_brier_score(y_true, probs_2d, score_type="binary"), bs_bin, places=6)
        self.assertAlmostEqual(compute_brier_score(y_true, probs_2d, score_type="multiclass"), bs_multi, places=6)

    def test_torch_tensor_and_list_support(self):
        """Verify seamless operation with PyTorch tensors and Python lists."""
        y_list = [0, 1, 1, 0]
        p_list = [0.1, 0.9, 0.8, 0.2]

        y_tensor = torch.tensor(y_list, dtype=torch.long)
        p_tensor = torch.tensor(p_list, dtype=torch.float32)

        metrics_list = compute_calibration_metrics(y_list, p_list, num_bins=5)
        metrics_tensor = compute_calibration_metrics(y_tensor, p_tensor, num_bins=5)

        self.assertAlmostEqual(
            metrics_list["brier_score_binary"], metrics_tensor["brier_score_binary"], places=5
        )
        self.assertAlmostEqual(
            metrics_list["negative_log_likelihood"], metrics_tensor["negative_log_likelihood"], places=5
        )
        self.assertAlmostEqual(
            metrics_list["ece_positive_class"], metrics_tensor["ece_positive_class"], places=5
        )
        self.assertAlmostEqual(
            metrics_list["ece_top_label"], metrics_tensor["ece_top_label"], places=5
        )

    def test_input_validation_errors(self):
        """Verify robust error raising on invalid inputs."""
        # Empty arrays
        with self.assertRaises(ValueError):
            compute_binary_brier_score([], [])

        # Sample count mismatch
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [0.5])

        # Non-binary labels
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 2], [0.5, 0.5])

        with self.assertRaises(ValueError):
            compute_binary_brier_score([-1, 1], [0.5, 0.5])

        # Probabilities < 0.0 or > 1.0
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [-0.1, 0.5])

        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [0.5, 1.2])

        # 2D probabilities not summing to 1.0
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [[0.3, 0.3], [0.5, 0.5]])

        # 2D probabilities wrong class count
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [[0.3, 0.3, 0.4], [0.3, 0.3, 0.4]])

        # Non-finite values
        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [np.nan, 0.5])

        with self.assertRaises(ValueError):
            compute_binary_brier_score([0, 1], [np.inf, 0.5])

        # Invalid num_bins
        with self.assertRaises(ValueError):
            compute_ece([0, 1], [0.5, 0.5], num_bins=1)

        # Invalid strategy
        with self.assertRaises(ValueError):
            compute_ece([0, 1], [0.5, 0.5], strategy="unknown_strategy")

        # Invalid score_type
        with self.assertRaises(ValueError):
            compute_brier_score([0, 1], [0.5, 0.5], score_type="unsupported")


if __name__ == "__main__":
    unittest.main()
