"""
Unit tests for TemperatureScaler post-hoc probability calibration utility.

Tests:
1. Default temperature before fitting is 1.0 and scaler is marked unfitted.
2. Fitting produces a finite, positive temperature within configured bounds.
3. Calibration reduces NLL on synthetic overconfident validation data.
4. Logit transformation strictly preserves predicted class (argmax invariance) when T > 0.
5. Softmax probabilities remain finite, bounded in [0, 1], and sum to 1.0.
6. Invalid shapes, labels, bounds, NaN, and Inf inputs raise descriptive ValueErrors.
7. Applying transform or predict_proba does not mutate input arrays/tensors.
8. REAL (index 0) and FAKE (index 1) output order and semantics are preserved.
"""

import unittest
import numpy as np
import torch

from src.models.temperature_scaling import TemperatureScaler
from src.evaluation.calibration import compute_nll


class TestTemperatureScaler(unittest.TestCase):
    """Test suite verifying TemperatureScaler mathematical behavior and safeguards."""

    def setUp(self):
        # Deterministic random generator for reproducible synthetic tests
        self.rng = np.random.RandomState(42)

    def test_default_unfitted_state(self):
        """Verifies default parameter values and identity scaling before fit."""
        scaler = TemperatureScaler()
        self.assertEqual(scaler.temperature, 1.0)
        self.assertFalse(scaler.is_fitted)
        self.assertIsNone(scaler.fitted_loss_)
        self.assertIsNone(scaler.uncalibrated_loss_)

        # Applying before fitting should scale by 1.0 (identity)
        test_logits = np.array([[2.0, -1.0], [-0.5, 1.5]])
        scaled = scaler.transform(test_logits)
        np.testing.assert_allclose(scaled, test_logits)

    def test_fitting_within_bounds(self):
        """Verifies fitting yields a finite positive temperature within [t_min, t_max]."""
        # Generate simple synthetic binary logits
        n_samples = 100
        y = self.rng.randint(0, 2, size=n_samples)
        # Moderate confidence
        logits = np.zeros((n_samples, 2))
        for i in range(n_samples):
            logits[i] = [-1.5, 1.5] if y[i] == 1 else [1.5, -1.5]

        scaler = TemperatureScaler(t_min=0.1, t_max=5.0)
        scaler.fit(logits, y)

        self.assertTrue(scaler.is_fitted)
        self.assertTrue(np.isfinite(scaler.temperature))
        self.assertGreaterEqual(scaler.temperature, 0.1)
        self.assertLessEqual(scaler.temperature, 5.0)
        self.assertIsNotNone(scaler.fitted_loss_)
        self.assertTrue(np.isfinite(scaler.fitted_loss_))

    def test_calibration_reduces_nll_on_overconfident_example(self):
        """
        Construct an overconfident synthetic validation set (80% accuracy, extreme logits ±6.0).
        Fitting temperature scaling must reduce cross-entropy NLL substantially.
        """
        n_samples = 300
        y = self.rng.binomial(1, 0.5, size=n_samples)
        logits = np.zeros((n_samples, 2))

        for i in range(n_samples):
            is_correct = self.rng.rand() < 0.80  # 80% accuracy
            if is_correct:
                logits[i] = [-6.0, 6.0] if y[i] == 1 else [6.0, -6.0]
            else:
                logits[i] = [6.0, -6.0] if y[i] == 1 else [-6.0, 6.0]

        scaler = TemperatureScaler(t_min=0.05, t_max=10.0)
        scaler.fit(logits, y)

        # Temperature should be > 1.0 to soften overconfident logits
        self.assertGreater(scaler.temperature, 1.0)

        # Calibrated NLL must be strictly lower than uncalibrated NLL (at T=1.0)
        self.assertLess(scaler.fitted_loss_, scaler.uncalibrated_loss_)

        # Also verify via external calibration evaluation module
        probs_uncal = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        probs_cal = scaler.predict_proba(logits)

        nll_uncal = compute_nll(y, probs_uncal)
        nll_cal = compute_nll(y, probs_cal)
        self.assertLess(nll_cal, nll_uncal)

    def test_argmax_invariance(self):
        """
        For any positive scalar T > 0, scaling logits by 1/T must strictly preserve
        the predicted class (argmax invariance).
        """
        n_samples = 150
        logits = self.rng.randn(n_samples, 2) * 5.0
        orig_preds = np.argmax(logits, axis=1)

        # Test both softening (T > 1) and sharpening (T < 1)
        for t_val in [0.2, 0.5, 1.0, 2.5, 8.0]:
            scaler = TemperatureScaler(initial_temperature=t_val)
            scaler.temperature = t_val

            scaled_preds = scaler.predict(logits)
            np.testing.assert_array_equal(
                scaled_preds,
                orig_preds,
                err_msg=f"Argmax predictions altered at temperature T={t_val}",
            )

    def test_probability_properties(self):
        """Verifies calibrated probabilities are finite, in [0, 1], and sum to 1.0."""
        logits_np = self.rng.randn(100, 2) * 4.0
        logits_torch = torch.from_numpy(logits_np).float()

        scaler = TemperatureScaler(initial_temperature=2.0)

        # NumPy input
        probs_np = scaler.predict_proba(logits_np)
        self.assertIsInstance(probs_np, np.ndarray)
        self.assertEqual(probs_np.shape, (100, 2))
        self.assertTrue(np.all(np.isfinite(probs_np)))
        self.assertTrue(np.all(probs_np >= 0.0))
        self.assertTrue(np.all(probs_np <= 1.0))
        np.testing.assert_allclose(np.sum(probs_np, axis=1), 1.0, atol=1e-6)

        # PyTorch input
        probs_torch = scaler.predict_proba(logits_torch)
        self.assertIsInstance(probs_torch, torch.Tensor)
        self.assertEqual(probs_torch.shape, (100, 2))
        self.assertTrue(torch.all(torch.isfinite(probs_torch)))
        self.assertTrue(torch.all(probs_torch >= 0.0))
        self.assertTrue(torch.all(probs_torch <= 1.0))
        np.testing.assert_allclose(probs_torch.sum(dim=-1).numpy(), 1.0, atol=1e-6)

    def test_input_mutation_safety(self):
        """Applying transform or predict_proba must never mutate the input arrays/tensors."""
        logits_np = np.array([[1.5, -2.5], [-0.5, 3.0]], dtype=np.float64)
        np_copy = logits_np.copy()

        logits_torch = torch.tensor([[1.5, -2.5], [-0.5, 3.0]], dtype=torch.float32)
        torch_copy = logits_torch.clone()

        scaler = TemperatureScaler(initial_temperature=2.5)

        _ = scaler.transform(logits_np)
        _ = scaler.predict_proba(logits_np)
        np.testing.assert_array_equal(logits_np, np_copy)

        _ = scaler.transform(logits_torch)
        _ = scaler.predict_proba(logits_torch)
        self.assertTrue(torch.equal(logits_torch, torch_copy))

    def test_real_fake_class_index_order(self):
        """
        Verifies that index 0 corresponds to REAL and index 1 corresponds to FAKE.
        Higher logit at index 0 must produce higher probability at index 0, and vice versa.
        """
        scaler = TemperatureScaler(initial_temperature=1.5)

        # Case 1: High REAL evidence (logit 0 > logit 1)
        real_dominated = np.array([[4.0, 1.0]])
        probs_real = scaler.predict_proba(real_dominated)
        self.assertGreater(probs_real[0, 0], probs_real[0, 1])  # P(REAL) > P(FAKE)
        self.assertEqual(scaler.predict(real_dominated)[0], 0)   # Predicted class is 0 (REAL)

        # Case 2: High FAKE evidence (logit 1 > logit 0)
        fake_dominated = np.array([[1.0, 4.0]])
        probs_fake = scaler.predict_proba(fake_dominated)
        self.assertGreater(probs_fake[0, 1], probs_fake[0, 0])  # P(FAKE) > P(REAL)
        self.assertEqual(scaler.predict(fake_dominated)[0], 1)   # Predicted class is 1 (FAKE)

    def test_input_validation_errors(self):
        """Verifies robust exception raising on invalid inputs, labels, and bounds."""
        scaler = TemperatureScaler()

        # Bounds validation in constructor
        with self.assertRaises(ValueError):
            TemperatureScaler(t_min=0.0)
        with self.assertRaises(ValueError):
            TemperatureScaler(t_min=-0.5)
        with self.assertRaises(ValueError):
            TemperatureScaler(t_min=5.0, t_max=2.0)
        with self.assertRaises(ValueError):
            TemperatureScaler(initial_temperature=-1.0)

        # Empty inputs
        with self.assertRaises(ValueError):
            scaler.fit([], [])
        with self.assertRaises(ValueError):
            scaler.transform([])

        # Invalid logits shapes
        with self.assertRaises(ValueError):
            scaler.transform(np.array([1.0, 2.0]))  # 1D instead of 2D
        with self.assertRaises(ValueError):
            scaler.transform(np.zeros((5, 3)))      # 3 classes instead of 2
        with self.assertRaises(ValueError):
            scaler.transform(np.zeros((5, 1)))      # 1 class instead of 2

        # Non-finite logits
        with self.assertRaises(ValueError):
            scaler.transform(np.array([[np.nan, 1.0]]))
        with self.assertRaises(ValueError):
            scaler.transform(np.array([[np.inf, 1.0]]))

        # Invalid labels in fit
        valid_logits = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            scaler.fit(valid_logits, [0])           # Length mismatch
        with self.assertRaises(ValueError):
            scaler.fit(valid_logits, [0, 2])        # Non-binary label 2
        with self.assertRaises(ValueError):
            scaler.fit(valid_logits, [-1, 1])       # Non-binary label -1
        with self.assertRaises(ValueError):
            scaler.fit(valid_logits, [np.nan, 0])   # Non-finite label


if __name__ == "__main__":
    unittest.main()
