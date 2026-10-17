"""
Probability Calibration Metrics for Misinformation Cascade Classification.

This module provides calibration metrics and reliability diagram computations
specifically designed for graph-level cascade veracity classification.

Verified Class Mapping:
    - Class 0: REAL (Negative class)
    - Class 1: FAKE (Positive class, target for cascade detection)

Metric Definitions & Important Distinctions:
1. Brier Score:
   - Binary Brier Score: Mean squared error of positive-class probability P(FAKE):
         BS_binary = (1 / N) * sum_{i=1}^N (p_{i, 1} - y_i)^2
     Bounded in [0, 1], where 0 represents perfect calibration and sharpness.
   - Multiclass Brier Score: Sum of squared errors across all K classes:
         BS_multi = (1 / N) * sum_{i=1}^N sum_{k=0}^{K-1} (p_{i, k} - I(y_i == k))^2
     For binary classification (K=2) with probabilities summing to 1:
         BS_multi = 2 * BS_binary (bounded in [0, 2]).
     We distinguish these two definitions explicitly to prevent ambiguity.

2. Negative Log-Likelihood (NLL / Cross-Entropy):
   - Measures probabilistic loss with numerical clipping safeguards (eps = 1e-15):
         NLL = - (1 / N) * sum_{i=1}^N log(p_{i, y_i})
   - Heavily penalizes over-confident incorrect predictions.

3. Calibration Error and Binning:
   - Positive-Class Calibration (Marginal Calibration for Class 1 = FAKE):
     Bins predicted P(FAKE) in [0, 1]. For each bin B_m:
         conf(B_m) = mean(p_{i, 1} for i in B_m)
         acc(B_m)  = fraction of true FAKE samples (y_i == 1) in B_m
     ECE_pos = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
   - Top-Label Confidence Calibration (Multi-Class / Confidence ECE):
     Bins max class confidence p_hat_i = max_k p_{i, k} in [0, 1]. For each bin B_m:
         conf(B_m) = mean(p_hat_i for i in B_m)
         acc(B_m)  = classification accuracy (I(y_hat_i == y_i)) in B_m
     ECE_top = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
   NOTE: For binary classification, max confidence is bounded in [0.5, 1.0], whereas
   positive-class probability spans [0.0, 1.0]. These metrics measure distinct properties
   and are NOT interchangeable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def _validate_and_normalize_inputs(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validates array dimensions, label values, probability ranges, and probability normalization.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities. Can be 1D of shape (N,) representing P(FAKE),
        or 2D of shape (N, 2) representing [P(REAL), P(FAKE)].

    Returns
    -------
    y_true_arr : np.ndarray
        1D int64 array of shape (N,) with values in {0, 1}.
    prob_fake : np.ndarray
        1D float64 array of shape (N,) with P(FAKE) in [0.0, 1.0].
    probs_2d : np.ndarray
        2D float64 array of shape (N, 2) where col 0 is P(REAL) and col 1 is P(FAKE),
        row sums equal 1.0.

    Raises
    ------
    ValueError
        If inputs are empty, dimensions mismatch, labels are non-binary,
        probabilities are outside [0, 1], or row sums do not equal 1.0.
    TypeError
        If input types cannot be converted to numeric arrays.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true_arr = np.asarray(y_true)
    y_probs_arr = np.asarray(y_probs, dtype=np.float64)

    # Validate non-empty
    if y_true_arr.size == 0 or y_probs_arr.size == 0:
        raise ValueError("y_true and y_probs must not be empty.")

    # Flatten y_true to 1D
    if y_true_arr.ndim > 1:
        if y_true_arr.ndim == 2 and (y_true_arr.shape[1] == 1 or y_true_arr.shape[0] == 1):
            y_true_arr = y_true_arr.flatten()
        else:
            raise ValueError(f"y_true must be 1D or squeezable to 1D, got shape {y_true_arr.shape}.")
    y_true_arr = y_true_arr.astype(np.int64)

    # Check finite
    if not np.all(np.isfinite(y_true_arr)):
        raise ValueError("y_true contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(y_probs_arr)):
        raise ValueError("y_probs contains non-finite values (NaN or Inf).")

    # Validate binary label values {0, 1}
    unique_labels = np.unique(y_true_arr)
    invalid_labels = set(unique_labels) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"y_true must only contain binary labels 0 (REAL) or 1 (FAKE). Found: {sorted(list(unique_labels))}"
        )

    n_samples = len(y_true_arr)

    # Validate y_probs shapes and ranges
    if y_probs_arr.ndim == 1:
        if len(y_probs_arr) != n_samples:
            raise ValueError(
                f"Sample count mismatch: y_true has {n_samples} samples, "
                f"y_probs has {len(y_probs_arr)}."
            )
        if np.any(y_probs_arr < 0.0) or np.any(y_probs_arr > 1.0):
            raise ValueError(
                f"Probabilities must be in [0.0, 1.0]. Range found: "
                f"[{y_probs_arr.min():.4f}, {y_probs_arr.max():.4f}]."
            )
        prob_fake = y_probs_arr
        prob_real = 1.0 - prob_fake
        probs_2d = np.column_stack([prob_real, prob_fake])

    elif y_probs_arr.ndim == 2:
        if y_probs_arr.shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch: y_true has {n_samples} samples, "
                f"y_probs has {y_probs_arr.shape[0]} rows."
            )
        if y_probs_arr.shape[1] != 2:
            raise ValueError(
                f"For 2D y_probs, exactly 2 classes [P(REAL), P(FAKE)] are expected, "
                f"got shape {y_probs_arr.shape}."
            )
        if np.any(y_probs_arr < 0.0) or np.any(y_probs_arr > 1.0):
            raise ValueError(
                f"Probabilities must be in [0.0, 1.0]. Range found: "
                f"[{y_probs_arr.min():.4f}, {y_probs_arr.max():.4f}]."
            )

        row_sums = np.sum(y_probs_arr, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            max_dev = np.max(np.abs(row_sums - 1.0))
            raise ValueError(
                f"Probabilities per sample must sum to 1.0 (atol=1e-3). "
                f"Maximum deviation found: {max_dev:.6f}."
            )
        probs_2d = y_probs_arr
        prob_fake = y_probs_arr[:, 1]

    else:
        raise ValueError(f"y_probs must be 1D or 2D, got ndim={y_probs_arr.ndim}.")

    return y_true_arr, prob_fake, probs_2d


def compute_binary_brier_score(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
) -> float:
    """
    Computes the standard binary Brier score for class 1 (FAKE).

    Formula:
        BS_binary = (1 / N) * sum_{i=1}^N (P(FAKE)_i - y_i)^2

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities. Shape (N,) of P(FAKE) or (N, 2) of [P(REAL), P(FAKE)].

    Returns
    -------
    float
        Binary Brier score in range [0.0, 1.0]. Lower is better; 0.0 is perfect.
    """
    y_true_arr, prob_fake, _ = _validate_and_normalize_inputs(y_true, y_probs)
    return float(np.mean((prob_fake - y_true_arr) ** 2))


def compute_multiclass_brier_score(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
) -> float:
    """
    Computes the multiclass Brier score across both classes [REAL, FAKE].

    Formula:
        BS_multi = (1 / N) * sum_{i=1}^N sum_{k=0}^1 (p_{i, k} - I(y_i == k))^2

    Note:
        For binary classification with normalized probabilities,
        BS_multi = 2 * BS_binary. Bounded in [0.0, 2.0].

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities. Shape (N,) of P(FAKE) or (N, 2) of [P(REAL), P(FAKE)].

    Returns
    -------
    float
        Multiclass Brier score in range [0.0, 2.0]. Lower is better; 0.0 is perfect.
    """
    y_true_arr, _, probs_2d = _validate_and_normalize_inputs(y_true, y_probs)
    n_samples = len(y_true_arr)

    # One-hot encode targets
    y_one_hot = np.zeros_like(probs_2d)
    y_one_hot[np.arange(n_samples), y_true_arr] = 1.0

    return float(np.mean(np.sum((probs_2d - y_one_hot) ** 2, axis=1)))


def compute_brier_score(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    score_type: str = "binary",
) -> float:
    """
    Calculates Brier score with explicit selection of score type.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities.
    score_type : str, default="binary"
        Either 'binary' (MSE on P(FAKE), range [0, 1]) or
        'multiclass' (sum of MSEs across both classes, range [0, 2]).

    Returns
    -------
    float
        Brier score under the chosen formulation.
    """
    score_type = score_type.strip().lower()
    if score_type == "binary":
        return compute_binary_brier_score(y_true, y_probs)
    elif score_type == "multiclass":
        return compute_multiclass_brier_score(y_true, y_probs)
    else:
        raise ValueError(
            f"Invalid score_type='{score_type}'. Must be either 'binary' or 'multiclass'."
        )


def compute_nll(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    eps: float = 1e-15,
) -> float:
    """
    Calculates Negative Log-Likelihood (NLL / Log Loss) from probabilities and labels.

    Formula:
        NLL = - (1 / N) * sum_{i=1}^N log(p_{i, y_i})

    Safeguards:
        Probabilities are clipped to [eps, 1.0 - eps] to prevent log(0) and numerical overflow.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities. Shape (N,) of P(FAKE) or (N, 2) of [P(REAL), P(FAKE)].
    eps : float, default=1e-15
        Numerical threshold to clamp probabilities.

    Returns
    -------
    float
        Negative log-likelihood in range [0.0, inf). Lower is better.
    """
    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}.")

    y_true_arr, _, probs_2d = _validate_and_normalize_inputs(y_true, y_probs)
    n_samples = len(y_true_arr)

    # Extract probability assigned to true class
    true_class_probs = probs_2d[np.arange(n_samples), y_true_arr]

    # Clip for numerical stability
    clipped_probs = np.clip(true_class_probs, eps, 1.0 - eps)

    nll = -np.mean(np.log(clipped_probs))
    return float(nll)


def compute_reliability_diagram_data(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    num_bins: int = 10,
    strategy: str = "positive_class",
) -> Dict[str, Any]:
    """
    Computes reliability diagram binning data and calibration error statistics.

    Binning Convention:
        - Bins divide the [0.0, 1.0] interval into `num_bins` uniform sub-intervals:
          bin m spans [m / num_bins, (m + 1) / num_bins).
        - The upper boundary of the final bin is closed: [(num_bins - 1) / num_bins, 1.0].
        - A sample with value exactly 1.0 is assigned to the final bin.

    Strategies:
        1. 'positive_class':
           Bins on the predicted probability of the positive class P(FAKE) in [0.0, 1.0].
           Within each bin:
               - sample_count: number of predictions in bin.
               - bin_confidence: mean P(FAKE) in bin.
               - bin_accuracy: empirical positive rate (fraction of true FAKE labels, y == 1).
               - bin_gap: |bin_accuracy - bin_confidence|.
           ECE is the sample-weighted average gap across non-empty bins.
        2. 'top_label':
           Bins on the model's top confidence max_k P(k) in [0.0, 1.0].
           Within each bin:
               - sample_count: number of predictions in bin.
               - bin_confidence: mean top confidence in bin.
               - bin_accuracy: empirical classification accuracy (I(y_pred == y_true)).
               - bin_gap: |bin_accuracy - bin_confidence|.
           ECE is the sample-weighted average gap across non-empty bins.
           NOTE: For binary classification, top confidence is always >= 0.5,
           so bins below 0.5 are typically empty.

    Empty Bins:
        If a bin contains 0 samples:
        - sample_count = 0
        - bin_confidence = np.nan
        - bin_accuracy = np.nan
        - bin_gap = np.nan
        - Empty bins contribute 0 weight to ECE and are excluded from MCE.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities.
    num_bins : int, default=10
        Number of equal-width bins in [0, 1]. Must be >= 2.
    strategy : str, default="positive_class"
        Either 'positive_class' or 'top_label'.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'strategy': str, evaluated calibration strategy
        - 'num_bins': int, number of bins
        - 'total_samples': int, N
        - 'bin_edges': np.ndarray of shape (num_bins + 1,)
        - 'bin_centers': np.ndarray of shape (num_bins,)
        - 'sample_counts': np.ndarray of shape (num_bins,) (int)
        - 'bin_confidences': np.ndarray of shape (num_bins,) (float or NaN)
        - 'bin_accuracies': np.ndarray of shape (num_bins,) (float or NaN)
        - 'bin_gaps': np.ndarray of shape (num_bins,) (float or NaN)
        - 'ece': float, Expected Calibration Error
        - 'mce': float, Maximum Calibration Error across non-empty bins
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be an integer >= 2, got {num_bins}.")

    strategy = strategy.strip().lower()
    valid_strategies = {"positive_class", "top_label"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid calibration strategy='{strategy}'. Must be one of {sorted(list(valid_strategies))}."
        )

    y_true_arr, prob_fake, probs_2d = _validate_and_normalize_inputs(y_true, y_probs)
    n_samples = len(y_true_arr)

    # Determine confidence values and correctness indicators based on strategy
    if strategy == "positive_class":
        # Calibrating P(FAKE): confidence is P(FAKE), target correctness is y_true == 1
        confidences = prob_fake
        targets = (y_true_arr == 1).astype(np.float64)
    else:  # 'top_label'
        # Calibrating predicted top class: confidence is max_k P(k), correctness is y_pred == y_true
        y_pred = np.argmax(probs_2d, axis=1)
        confidences = np.max(probs_2d, axis=1)
        targets = (y_pred == y_true_arr).astype(np.float64)

    # Uniform bin edges in [0.0, 1.0]
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Assign samples to bins:
    # bin_edges[1:-1] provides (num_bins - 1) internal boundaries.
    # np.digitize with right=False gives:
    #   conf < bin_edges[1] => 0
    #   bin_edges[1] <= conf < bin_edges[2] => 1
    #   ...
    #   conf >= bin_edges[-2] => num_bins - 1
    # Confidences clamped to [0.0, 1.0] ensure indices in {0, ..., num_bins - 1}.
    clamped_confs = np.clip(confidences, 0.0, 1.0)
    bin_indices = np.digitize(clamped_confs, bin_edges[1:-1], right=False)

    sample_counts = np.zeros(num_bins, dtype=np.int64)
    bin_confidences = np.full(num_bins, np.nan, dtype=np.float64)
    bin_accuracies = np.full(num_bins, np.nan, dtype=np.float64)
    bin_gaps = np.full(num_bins, np.nan, dtype=np.float64)

    ece = 0.0
    valid_gaps = []

    for b in range(num_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        sample_counts[b] = count

        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(targets[mask]))
            gap = abs(avg_acc - avg_conf)

            bin_confidences[b] = avg_conf
            bin_accuracies[b] = avg_acc
            bin_gaps[b] = gap

            ece += (count / n_samples) * gap
            valid_gaps.append(gap)

    mce = float(max(valid_gaps)) if valid_gaps else 0.0

    return {
        "strategy": strategy,
        "num_bins": num_bins,
        "total_samples": n_samples,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "sample_counts": sample_counts,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_gaps": bin_gaps,
        "ece": float(ece),
        "mce": mce,
    }


def compute_ece(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    num_bins: int = 10,
    strategy: str = "positive_class",
) -> float:
    """
    Computes Expected Calibration Error (ECE) for either positive-class or top-label calibration.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities.
    num_bins : int, default=10
        Number of calibration bins.
    strategy : str, default="positive_class"
        Either 'positive_class' (calibration of P(FAKE) over [0, 1]) or
        'top_label' (calibration of max confidence over [0, 1]).

    Returns
    -------
    float
        ECE score in range [0.0, 1.0]. Lower is better; 0.0 is perfectly calibrated.
    """
    data = compute_reliability_diagram_data(
        y_true=y_true, y_probs=y_probs, num_bins=num_bins, strategy=strategy
    )
    return float(data["ece"])


def compute_positive_class_ece(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    num_bins: int = 10,
) -> float:
    """
    Explicit convenience function for positive-class (P(FAKE)) calibration error.

    Measures calibration of predicted probability of the misinformation class:
        ECE_pos = sum_{m=1}^M (|B_m| / N) * |empirical_fake_rate(B_m) - mean_prob_fake(B_m)|
    """
    return compute_ece(y_true=y_true, y_probs=y_probs, num_bins=num_bins, strategy="positive_class")


def compute_top_label_ece(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    num_bins: int = 10,
) -> float:
    """
    Explicit convenience function for top-label confidence calibration error.

    Measures calibration of model confidence in its chosen prediction:
        ECE_top = sum_{m=1}^M (|B_m| / N) * |accuracy(B_m) - mean_confidence(B_m)|
    """
    return compute_ece(y_true=y_true, y_probs=y_probs, num_bins=num_bins, strategy="top_label")


def compute_calibration_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Union[List[float], List[List[float]], np.ndarray, torch.Tensor],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Comprehensive calibration evaluation suite returning all scalar metrics
    and full reliability diagram curves.

    Parameters
    ----------
    y_true : Union[List[int], np.ndarray, torch.Tensor]
        Ground truth binary labels (0 = REAL, 1 = FAKE).
    y_probs : Union[List, np.ndarray, torch.Tensor]
        Predicted probabilities.
    num_bins : int, default=10
        Number of bins for ECE and reliability diagrams.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'brier_score_binary': float, binary Brier score in [0, 1]
        - 'brier_score_multiclass': float, multiclass Brier score in [0, 2]
        - 'negative_log_likelihood': float, cross-entropy NLL >= 0
        - 'ece_positive_class': float, ECE for P(FAKE)
        - 'mce_positive_class': float, MCE for P(FAKE)
        - 'ece_top_label': float, ECE for max confidence
        - 'mce_top_label': float, MCE for max confidence
        - 'reliability_positive_class': Dict, reliability diagram curve for P(FAKE)
        - 'reliability_top_label': Dict, reliability diagram curve for top confidence
    """
    brier_bin = compute_binary_brier_score(y_true, y_probs)
    brier_multi = compute_multiclass_brier_score(y_true, y_probs)
    nll = compute_nll(y_true, y_probs)

    rel_pos = compute_reliability_diagram_data(
        y_true, y_probs, num_bins=num_bins, strategy="positive_class"
    )
    rel_top = compute_reliability_diagram_data(
        y_true, y_probs, num_bins=num_bins, strategy="top_label"
    )

    return {
        "brier_score_binary": brier_bin,
        "brier_score_multiclass": brier_multi,
        "negative_log_likelihood": nll,
        "ece_positive_class": rel_pos["ece"],
        "mce_positive_class": rel_pos["mce"],
        "ece_top_label": rel_top["ece"],
        "mce_top_label": rel_top["mce"],
        "reliability_positive_class": rel_pos,
        "reliability_top_label": rel_top,
    }
