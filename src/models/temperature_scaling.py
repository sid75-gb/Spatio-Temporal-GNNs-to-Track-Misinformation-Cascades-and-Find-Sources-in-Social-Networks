"""
Post-hoc Probability Calibration via Temperature Scaling.

This module implements Platt / Temperature Scaling (Guo et al., 2017) for binary
cascade veracity classification.

CRITICAL METHODOLOGICAL REQUIREMENT:
    The temperature scalar T must be fitted exclusively on VALIDATION split logits
    and VALIDATION split labels. The test split must NEVER be accessed or used during
    temperature optimization to prevent data leakage and invalid evaluation results.

Class Mapping & Mathematical Properties:
    - Index 0: REAL (Negative class)
    - Index 1: FAKE (Positive class)

    Logit Transformation:
        z_scaled = z / T
        P(REAL) = exp(z_0 / T) / (exp(z_0 / T) + exp(z_1 / T))
        P(FAKE) = exp(z_1 / T) / (exp(z_0 / T) + exp(z_1 / T))

    Monotonicity & Invariance:
        For any positive scalar T > 0:
            z_0 > z_1 <=> z_0 / T > z_1 / T
        - The predicted class argmax(z) is strictly invariant under temperature scaling.
        - Binary decision threshold at probability 0.5 is unchanged.
        - Classification accuracy, confusion matrix, and ROC-AUC are preserved exactly.
        - Cross-entropy loss (NLL), Brier score, and calibration errors (ECE, MCE)
          are optimized without modifying model parameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch


class TemperatureScaler:
    """
    Post-hoc probability calibration via bounded Temperature Scaling.

    Learns a single positive scalar T by minimizing Cross-Entropy / Negative
    Log-Likelihood (NLL) on validation logits.

    Parameters
    ----------
    t_min : float, default=0.05
        Lower bound for scalar temperature search. Must be > 0.
    t_max : float, default=10.0
        Upper bound for scalar temperature search. Must be > t_min.
    initial_temperature : float, default=1.0
        Default temperature parameter before fitting.
    """

    def __init__(
        self,
        t_min: float = 0.05,
        t_max: float = 10.0,
        initial_temperature: float = 1.0,
    ) -> None:
        if t_min <= 0.0:
            raise ValueError(f"t_min must be strictly positive (> 0), got {t_min}.")
        if t_max <= t_min:
            raise ValueError(f"t_max must be greater than t_min, got t_min={t_min}, t_max={t_max}.")
        if initial_temperature <= 0.0:
            raise ValueError(
                f"initial_temperature must be strictly positive, got {initial_temperature}."
            )

        self.t_min: float = float(t_min)
        self.t_max: float = float(t_max)
        self.temperature: float = float(initial_temperature)
        self.is_fitted: bool = False
        self.fitted_loss_: Optional[float] = None
        self.uncalibrated_loss_: Optional[float] = None

    def _validate_inputs(
        self,
        logits: Union[np.ndarray, torch.Tensor, List],
        labels: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        """
        Validates shapes, types, ranges, and non-finite values for logits and optional labels.

        Returns
        -------
        logits_arr : np.ndarray
            2D float64 array of shape [N, 2].
        labels_arr : Optional[np.ndarray]
            1D int64 array of shape [N] with values in {0, 1}.
        is_torch : bool
            True if primary input was a torch.Tensor.
        """
        is_torch = isinstance(logits, torch.Tensor)

        # Convert logits
        if is_torch:
            logits_arr = logits.detach().cpu().numpy()
        else:
            logits_arr = np.asarray(logits, dtype=np.float64)

        if logits_arr.size == 0:
            raise ValueError("Logits array must not be empty.")

        if logits_arr.ndim != 2 or logits_arr.shape[1] != 2:
            raise ValueError(
                f"Logits must be 2D with shape [N, 2], got shape {logits_arr.shape}."
            )

        if not np.all(np.isfinite(logits_arr)):
            raise ValueError("Logits contain non-finite values (NaN, +Inf, or -Inf).")

        n_samples = logits_arr.shape[0]

        # Convert and validate labels if provided
        labels_arr = None
        if labels is not None:
            if isinstance(labels, torch.Tensor):
                labels_arr = labels.detach().cpu().numpy()
            else:
                labels_arr = np.asarray(labels)

            if labels_arr.size == 0:
                raise ValueError("Labels array must not be empty.")

            if labels_arr.ndim > 1:
                if labels_arr.ndim == 2 and (labels_arr.shape[1] == 1 or labels_arr.shape[0] == 1):
                    labels_arr = labels_arr.flatten()
                else:
                    raise ValueError(
                        f"Labels must be 1D or squeezable to 1D, got shape {labels_arr.shape}."
                    )

            if len(labels_arr) != n_samples:
                raise ValueError(
                    f"Sample count mismatch: logits has {n_samples} samples, "
                    f"labels has {len(labels_arr)}."
                )

            if not np.all(np.isfinite(labels_arr)):
                raise ValueError("Labels contain non-finite values (NaN, +Inf, or -Inf).")

            labels_arr = labels_arr.astype(np.int64)
            unique_labels = set(np.unique(labels_arr))
            invalid_labels = unique_labels - {0, 1}
            if invalid_labels:
                raise ValueError(
                    f"Labels must only contain binary values 0 (REAL) or 1 (FAKE). "
                    f"Found invalid labels: {sorted(list(invalid_labels))}."
                )

        return logits_arr, labels_arr, is_torch

    def fit(
        self,
        logits: Union[np.ndarray, torch.Tensor, List],
        labels: Union[np.ndarray, torch.Tensor, List],
    ) -> "TemperatureScaler":
        """
        Fits scalar temperature T on validation logits and labels by minimizing NLL.

        IMPORTANT:
        Fit this scaler ONLY on validation data. Do NOT pass test split logits or labels.

        Parameters
        ----------
        logits : Union[np.ndarray, torch.Tensor, List]
            Validation logits of shape [N, 2].
        labels : Union[np.ndarray, torch.Tensor, List]
            Validation ground truth binary labels of shape [N] (0 = REAL, 1 = FAKE).

        Returns
        -------
        self : TemperatureScaler
            The fitted scaler instance.
        """
        logits_arr, labels_arr, _ = self._validate_inputs(logits, labels)
        assert labels_arr is not None

        n_samples = len(labels_arr)
        sample_indices = np.arange(n_samples)

        def nll_objective(t_candidate: float) -> float:
            scaled = logits_arr / t_candidate
            # Numerically stable cross-entropy: logsumexp(z) - z[target]
            log_z = logsumexp(scaled, axis=1)
            target_logits = scaled[sample_indices, labels_arr]
            return float(np.mean(log_z - target_logits))

        # Compute initial uncalibrated loss at T=1.0
        self.uncalibrated_loss_ = nll_objective(1.0)

        # Bounded scalar optimization (Brent's method)
        opt_res = minimize_scalar(
            nll_objective,
            bounds=(self.t_min, self.t_max),
            method="bounded",
            options={"xatol": 1e-5, "maxiter": 500},
        )

        if not opt_res.success:
            # Fallback check: if optimization failed, keep default or warn
            raise RuntimeError(
                f"Temperature optimization failed to converge: {opt_res.message}"
            )

        self.temperature = float(opt_res.x)
        self.fitted_loss_ = float(opt_res.fun)
        self.is_fitted = True

        return self

    def transform(
        self,
        logits: Union[np.ndarray, torch.Tensor, List],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scales logits by dividing by the temperature T: z_scaled = z / T.

        Parameters
        ----------
        logits : Union[np.ndarray, torch.Tensor, List]
            Raw uncalibrated logits of shape [N, 2].

        Returns
        -------
        scaled_logits : Union[np.ndarray, torch.Tensor]
            Scaled logits of shape [N, 2]. Returns torch.Tensor if input was torch.Tensor,
            otherwise np.ndarray. The original input is never mutated.
        """
        logits_arr, _, is_torch = self._validate_inputs(logits, labels=None)

        if is_torch and isinstance(logits, torch.Tensor):
            # Non-mutating division on original torch device and dtype
            return logits / self.temperature

        # Non-mutating numpy division
        return logits_arr / self.temperature

    def predict_proba(
        self,
        logits: Union[np.ndarray, torch.Tensor, List],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calculates calibrated softmax probabilities from raw logits.

        Output shape is [N, 2] where:
            - Column 0: P(REAL) in [0.0, 1.0]
            - Column 1: P(FAKE) in [0.0, 1.0]
            - P(REAL) + P(FAKE) = 1.0

        Parameters
        ----------
        logits : Union[np.ndarray, torch.Tensor, List]
            Raw uncalibrated logits of shape [N, 2].

        Returns
        -------
        probs : Union[np.ndarray, torch.Tensor]
            Calibrated probabilities of shape [N, 2].
        """
        scaled = self.transform(logits)

        if isinstance(scaled, torch.Tensor):
            probs = torch.softmax(scaled, dim=-1)
            return probs

        # Numerically stable softmax in NumPy
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs

    def predict(
        self,
        logits: Union[np.ndarray, torch.Tensor, List],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Predicts binary class labels (0 = REAL, 1 = FAKE) from raw logits.

        Note:
            Because T > 0 is a strictly positive scalar transformation,
            predict(logits) is mathematically identical to argmax(logits) before scaling.
        """
        probs = self.predict_proba(logits)
        if isinstance(probs, torch.Tensor):
            return torch.argmax(probs, dim=-1)
        return np.argmax(probs, axis=1)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        loss_info = f", nll={self.fitted_loss_:.4f}" if self.fitted_loss_ is not None else ""
        return (
            f"TemperatureScaler(temperature={self.temperature:.4f}, status={status}{loss_info}, "
            f"bounds=[{self.t_min}, {self.t_max}])"
        )
