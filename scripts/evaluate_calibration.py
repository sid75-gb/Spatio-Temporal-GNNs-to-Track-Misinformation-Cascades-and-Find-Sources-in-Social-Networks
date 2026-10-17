"""
Standalone Probability Calibration Evaluation Script for Spatio-Temporal GNNs.

Evaluates post-hoc probability calibration (Temperature Scaling) on trained
Spatio-Temporal Graph Neural Network (STGNN) checkpoints for UPFD PolitiFact and GossipCop.

Strict Calibration Protocol:
1. Load existing trained checkpoint without retraining or modifying weights.
2. Extract validation logits and ground-truth labels sequentially from val split.
3. Fit TemperatureScaler exclusively on validation data (minimizing NLL).
4. Freeze the fitted temperature T*.
5. Evaluate uncalibrated and calibrated probabilities on the official held-out test split.
6. Verify exact reproduction of baseline test accuracy, ROC-AUC, and confusion matrix.
7. Compute full calibration suite: Brier score (binary & multiclass), NLL,
   Top-Label ECE, and Positive-Class ECE for P(FAKE) with reliability diagram data.
8. Safely save results to outputs/calibration_{dataset}.json without overwriting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from src.data.dataset_loader import load_upfd_data
from src.data.temporal_builder import TemporalCascadeBuilder
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.metrics import compute_classification_metrics
from src.models.stgnn import SpatioTemporalGNN
from src.models.temperature_scaling import TemperatureScaler


# Default checkpoint and baseline paths
DEFAULT_CHECKPOINTS: Dict[str, Path] = {
    "politifact": (
        PROJECT_ROOT
        / "checkpoints"
        / "politifact_timestamp_lr001_es8"
        / "stgnn_politifact_best.pt"
    ),
    "gossipcop": (
        PROJECT_ROOT
        / "checkpoints"
        / "gossipcop_timestamp_lr001_es8"
        / "stgnn_gossipcop_best.pt"
    ),
}

BASELINE_RESULT_PATHS: Dict[str, Path] = {
    "politifact": (
        PROJECT_ROOT
        / "outputs"
        / "stgnn_politifact_timestamp_lr001_es8.json"
    ),
    "gossipcop": (
        PROJECT_ROOT
        / "outputs"
        / "stgnn_gossipcop_timestamp_lr001_es8.json"
    ),
}


def sanitize_for_json(val: Any) -> Any:
    """
    Recursively sanitizes values for standards-compliant JSON serialization:
    - Converts NaN and Inf to null (None in Python).
    - Converts NumPy/PyTorch types to standard Python primitives.
    """
    if val is None:
        return None
    if isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        return f_val if np.isfinite(f_val) else None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, np.ndarray):
        return [sanitize_for_json(x) for x in val.tolist()]
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            item = val.item()
            if isinstance(item, float):
                return item if np.isfinite(item) else None
            return item
        return [sanitize_for_json(x) for x in val.detach().cpu().tolist()]
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_for_json(x) for x in val]
    return str(val)


def load_stgnn_model(
    checkpoint_path: Path,
    in_dim: int,
    device: torch.device,
) -> Tuple[SpatioTemporalGNN, Dict[str, Any]]:
    """
    Loads trained SpatioTemporalGNN weights from checkpoint.

    Validates model dimensions:
    Expects (in_dim + 4) input features when temporal snapshot features are attached.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_dim = int(checkpoint.get("hidden_dim", 64))

    # Checkpoint weight shape verification
    state_dict = checkpoint["model_state_dict"]
    expected_in_features = in_dim + 4
    actual_in_features = state_dict["input_proj.weight"].shape[1]
    if actual_in_features != expected_in_features:
        raise ValueError(
            f"Checkpoint input dimension mismatch: model expects {expected_in_features} "
            f"(in_dim={in_dim} + 4 temporal), but checkpoint has {actual_in_features}."
        )

    model = SpatioTemporalGNN(
        in_features=expected_in_features,
        hidden_dim=hidden_dim,
        num_snapshots=0,  # dynamic temporal length support
        temporal_cell="gru",
        num_classes=2,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def extract_split_logits_and_labels(
    dataset: Any,
    model: SpatioTemporalGNN,
    temporal_builder: TemporalCascadeBuilder,
    device: torch.device,
    split_name: str = "val",
    log_interval: int = 100,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Sequentially extracts raw model logits [N, 2] and true labels [N]
    for all cascades in a split at 100% full observation.
    """
    n_total = len(dataset)
    logits_list: List[np.ndarray] = []
    labels_list: List[int] = []
    skipped = 0
    t0 = time.time()

    print(f"  Extracting {split_name} split ({n_total} cascades)...")

    for idx in range(n_total):
        cascade = dataset[idx]

        try:
            snapshots = temporal_builder.build_snapshots(cascade)
        except Exception as exc:
            skipped += 1
            continue

        if len(snapshots) != 4:
            skipped += 1
            continue

        prefix_snapshots = [s.to(device) for s in snapshots]

        with torch.no_grad():
            raw_logits, _, _ = model.forward_snapshots(prefix_snapshots)
            logits_np = raw_logits.cpu().numpy()[0]

        logits_list.append(logits_np)
        labels_list.append(int(cascade.y.item()))

        if (idx + 1) % log_interval == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"    [{split_name.upper()}] [{idx + 1:5d}/{n_total:5d}] "
                f"Extracted: {len(logits_list):5d} | Speed: {rate:5.1f} casc/s",
                flush=True,
            )

    logits_arr = np.array(logits_list, dtype=np.float64)
    labels_arr = np.array(labels_list, dtype=np.int64)

    return logits_arr, labels_arr, len(logits_list), skipped


def evaluate_dataset_calibration(
    dataset_name: str,
    checkpoint_path: Path,
    data_root: str = "./data/UPFD",
    device_str: str = "cpu",
    output_dir: str = "./outputs",
    num_bins: int = 10,
    log_interval: int = 100,
) -> Optional[Dict[str, Any]]:
    """
    Executes full probability calibration evaluation for a dataset.
    """
    dataset_name = dataset_name.lower()
    device = torch.device(device_str)

    print("\n" + "=" * 75)
    print(f"  PROBABILITY CALIBRATION AUDIT & EVALUATION: UPFD-{dataset_name.upper()}")
    print("=" * 75)

    # Check destination file existence constraint: do NOT overwrite
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir_path / f"calibration_{dataset_name}.json"
    if out_file_path.exists():
        print(
            f"\n[ABORTED] Destination output file already exists:\n"
            f"  {out_file_path.resolve()}\n"
            f"Strict safety constraint: Existing calibration output files must NOT be overwritten.\n"
            f"Please rename or back up the existing file to proceed."
        )
        return None

    # 1. Load dataset with exact profile normalization
    print(f"\n[Step 1/6] Loading UPFD-{dataset_name.capitalize()} dataset splits...")
    train_ds, val_ds, test_ds, meta = load_upfd_data(
        root=data_root,
        name=dataset_name,
        feature="bert",
        multimodal=True,
    )
    in_dim = int(meta["input_dim"])
    print(
        f"  Loaded splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} | "
        f"Base Feature Dim={in_dim}"
    )

    # 2. Load model checkpoint
    print(f"\n[Step 2/6] Loading STGNN model checkpoint: {checkpoint_path.name}...")
    model, ckpt_meta = load_stgnn_model(
        checkpoint_path=checkpoint_path,
        in_dim=in_dim,
        device=device,
    )
    print(
        f"  Checkpoint successfully loaded (trained epoch: {ckpt_meta.get('epoch', 'N/A')}, "
        f"val_f1: {ckpt_meta.get('val_f1', 0.0):.4f})."
    )

    # Setup temporal builder (4 snapshots, timestamps required)
    temporal_builder = TemporalCascadeBuilder(
        num_snapshots=4,
        add_temporal_features=True,
        require_complete_timestamps=True,
    )

    # 3. Extract validation split logits & labels ONLY
    print(f"\n[Step 3/6] Extracting VALIDATION split logits for temperature fitting...")
    val_logits, val_labels, n_val_eval, n_val_skip = extract_split_logits_and_labels(
        dataset=val_ds,
        model=model,
        temporal_builder=temporal_builder,
        device=device,
        split_name="val",
        log_interval=log_interval,
    )
    print(
        f"  Validation data extracted: {n_val_eval} cascades "
        f"(Real: {np.sum(val_labels == 0)}, Fake: {np.sum(val_labels == 1)}). Skipped: {n_val_skip}."
    )

    # 4. Fit TemperatureScaler exclusively on validation data
    print(f"\n[Step 4/6] Fitting TemperatureScaler on VALIDATION data (zero test leakage)...")
    scaler = TemperatureScaler(t_min=0.05, t_max=10.0)
    scaler.fit(val_logits, val_labels)
    fitted_temp = scaler.temperature
    val_nll_uncal = float(scaler.uncalibrated_loss_)
    val_nll_cal = float(scaler.fitted_loss_)
    val_nll_delta = val_nll_uncal - val_nll_cal

    print(f"  Fitted Temperature T*  : {fitted_temp:.6f} (bounded in [0.05, 10.0])")
    print(f"  Validation NLL (T=1.0) : {val_nll_uncal:.6f}")
    print(f"  Validation NLL (T=T*)  : {val_nll_cal:.6f} (Reduction: {val_nll_delta:+.6f})")

    # 5. Extract held-out test split logits & labels
    print(f"\n[Step 5/6] Extracting TEST split logits for evaluation...")
    test_logits, test_labels, n_test_eval, n_test_skip = extract_split_logits_and_labels(
        dataset=test_ds,
        model=model,
        temporal_builder=temporal_builder,
        device=device,
        split_name="test",
        log_interval=log_interval if dataset_name == "politifact" else max(log_interval, 200),
    )
    print(
        f"  Test data extracted: {n_test_eval} cascades "
        f"(Real: {np.sum(test_labels == 0)}, Fake: {np.sum(test_labels == 1)}). Skipped: {n_test_skip}."
    )

    # 6. Evaluate uncalibrated test predictions
    uncal_probs = F.softmax(torch.from_numpy(test_logits), dim=-1).numpy()
    uncal_preds = np.argmax(uncal_probs, axis=1)

    uncal_class_metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=uncal_preds,
        y_probs=uncal_probs,
    )
    uncal_calib_metrics = compute_calibration_metrics(
        y_true=test_labels,
        y_probs=uncal_probs,
        num_bins=num_bins,
    )

    # Reproducibility verification check against saved benchmark
    baseline_path = BASELINE_RESULT_PATHS.get(dataset_name)
    if baseline_path and baseline_path.exists():
        print(f"\n--- Baseline Reproducibility Verification ---")
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base_metrics = base_data.get("metrics", {})

        diff_acc = abs(uncal_class_metrics["accuracy"] - base_metrics.get("accuracy", 0.0))
        diff_auc = abs(uncal_class_metrics["roc_auc"] - base_metrics.get("roc_auc", 0.0))
        cm_match = (
            uncal_class_metrics["tp"] == base_metrics.get("tp")
            and uncal_class_metrics["fp"] == base_metrics.get("fp")
            and uncal_class_metrics["tn"] == base_metrics.get("tn")
            and uncal_class_metrics["fn"] == base_metrics.get("fn")
        )

        print(
            f"  Expected Acc: {base_metrics.get('accuracy', 0.0):.6f} | "
            f"Obtained Acc: {uncal_class_metrics['accuracy']:.6f} (Diff: {diff_acc:.2e})"
        )
        print(
            f"  Expected AUC: {base_metrics.get('roc_auc', 0.0):.6f} | "
            f"Obtained AUC: {uncal_class_metrics['roc_auc']:.6f} (Diff: {diff_auc:.2e})"
        )
        print(
            f"  Confusion Matrix Match: {cm_match} "
            f"(TP={uncal_class_metrics['tp']}, FP={uncal_class_metrics['fp']}, "
            f"TN={uncal_class_metrics['tn']}, FN={uncal_class_metrics['fn']})"
        )

        if diff_acc > 1e-4 or diff_auc > 1e-4 or not cm_match:
            print(
                f"\n[REPRODUCIBILITY FAILURE] Obtained test predictions do not match baseline results:\n"
                f"  Baseline file: {baseline_path}\n"
                f"  Stopping evaluation for {dataset_name} without writing output file."
            )
            return None
        print("  [VERIFIED] Exact numerical reproducibility confirmed against saved benchmark.")
    else:
        print(f"  [NOTE] Baseline result file not found at {baseline_path}. Proceeding.")

    # 7. Evaluate calibrated test predictions
    cal_scaled_logits = scaler.transform(test_logits)
    cal_probs = scaler.predict_proba(test_logits)
    cal_preds = scaler.predict(test_logits)

    cal_class_metrics = compute_classification_metrics(
        y_true=test_labels,
        y_pred=cal_preds,
        y_probs=cal_probs,
    )
    cal_calib_metrics = compute_calibration_metrics(
        y_true=test_labels,
        y_probs=cal_probs,
        num_bins=num_bins,
    )

    # Invariance verification: argmax predictions, accuracy, and ROC-AUC must be invariant
    preds_identical = bool(np.array_equal(uncal_preds, cal_preds))
    acc_identical = bool(abs(uncal_class_metrics["accuracy"] - cal_class_metrics["accuracy"]) < 1e-6)
    auc_identical = bool(abs(uncal_class_metrics["roc_auc"] - cal_class_metrics["roc_auc"]) < 1e-6)

    print("\n--- Theoretical Invariance Verification ---")
    print(f"  Class Predictions Invariant (cal_preds == uncal_preds): {preds_identical}")
    print(f"  Test Accuracy Invariance                            : {acc_identical}")
    print(f"  Test ROC-AUC Invariance                             : {auc_identical}")
    assert preds_identical, "Temperature scaling altered discrete predicted classes!"
    assert acc_identical, "Temperature scaling altered classification accuracy!"

    # 8. Comparison summary
    summary_table = [
        ["Metric", "Uncalibrated (T=1.0)", f"Calibrated (T={fitted_temp:.4f})", "Delta"],
        ["Test Accuracy", f"{uncal_class_metrics['accuracy']:.4f}", f"{cal_class_metrics['accuracy']:.4f}", "0.0000 (Invariant)"],
        ["Macro F1", f"{uncal_class_metrics['f1_macro']:.4f}", f"{cal_class_metrics['f1_macro']:.4f}", "0.0000 (Invariant)"],
        ["Binary F1", f"{uncal_class_metrics['f1_binary']:.4f}", f"{cal_class_metrics['f1_binary']:.4f}", "0.0000 (Invariant)"],
        ["ROC-AUC", f"{uncal_class_metrics['roc_auc']:.4f}", f"{cal_class_metrics['roc_auc']:.4f}", "0.0000 (Invariant)"],
        ["Negative Log-Likelihood (NLL)", f"{uncal_calib_metrics['negative_log_likelihood']:.4f}", f"{cal_calib_metrics['negative_log_likelihood']:.4f}", f"{cal_calib_metrics['negative_log_likelihood'] - uncal_calib_metrics['negative_log_likelihood']:+.4f}"],
        ["Binary Brier Score (P(FAKE))", f"{uncal_calib_metrics['brier_score_binary']:.4f}", f"{cal_calib_metrics['brier_score_binary']:.4f}", f"{cal_calib_metrics['brier_score_binary'] - uncal_calib_metrics['brier_score_binary']:+.4f}"],
        ["Multiclass Brier Score", f"{uncal_calib_metrics['brier_score_multiclass']:.4f}", f"{cal_calib_metrics['brier_score_multiclass']:.4f}", f"{cal_calib_metrics['brier_score_multiclass'] - uncal_calib_metrics['brier_score_multiclass']:+.4f}"],
        ["Top-Label ECE", f"{uncal_calib_metrics['ece_top_label']:.4f}", f"{cal_calib_metrics['ece_top_label']:.4f}", f"{cal_calib_metrics['ece_top_label'] - uncal_calib_metrics['ece_top_label']:+.4f}"],
        ["Positive-Class ECE (P(FAKE))", f"{uncal_calib_metrics['ece_positive_class']:.4f}", f"{cal_calib_metrics['ece_positive_class']:.4f}", f"{cal_calib_metrics['ece_positive_class'] - uncal_calib_metrics['ece_positive_class']:+.4f}"],
    ]

    print("\n" + "=" * 75)
    print(f"  CALIBRATION RESULTS SUMMARY: {dataset_name.upper()}")
    print("=" * 75)
    if HAS_TABULATE:
        print(tabulate(summary_table[1:], headers=summary_table[0], tablefmt="grid"))
    else:
        for row in summary_table:
            print(f"  {row[0]:<32} | {row[1]:<20} | {row[2]:<22} | {row[3]}")
    print("=" * 75)

    # 9. Build structured results dictionary
    results_dict: Dict[str, Any] = {
        "dataset": dataset_name,
        "model": "stgnn",
        "observation_stage": "100% full observation",
        "num_snapshots": 4,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "temporal_mode": "timestamp",
        "class_mapping": {"0": "REAL", "1": "FAKE"},
        "calibration_protocol": {
            "method": "Temperature Scaling (Platt scaling)",
            "optimizer": "scipy.optimize.minimize_scalar (bounded Brent)",
            "fitted_on": "validation split logits and labels strictly (zero test leakage)",
            "temperature_bounds": [scaler.t_min, scaler.t_max],
            "fitted_temperature": fitted_temp,
        },
        "sample_counts": {
            "validation_total": n_val_eval,
            "validation_real": int(np.sum(val_labels == 0)),
            "validation_fake": int(np.sum(val_labels == 1)),
            "test_total": n_test_eval,
            "test_real": int(np.sum(test_labels == 0)),
            "test_fake": int(np.sum(test_labels == 1)),
        },
        "validation_metrics": {
            "uncalibrated_nll": val_nll_uncal,
            "calibrated_nll": val_nll_cal,
            "nll_reduction": val_nll_delta,
        },
        "test_metrics_uncalibrated": {
            "accuracy": uncal_class_metrics["accuracy"],
            "precision_macro": uncal_class_metrics["precision_macro"],
            "recall_macro": uncal_class_metrics["recall_macro"],
            "f1_macro": uncal_class_metrics["f1_macro"],
            "f1_binary": uncal_class_metrics["f1_binary"],
            "roc_auc": uncal_class_metrics["roc_auc"],
            "confusion_matrix": {
                "tp": uncal_class_metrics["tp"],
                "fp": uncal_class_metrics["fp"],
                "tn": uncal_class_metrics["tn"],
                "fn": uncal_class_metrics["fn"],
            },
            "brier_score_binary": uncal_calib_metrics["brier_score_binary"],
            "brier_score_multiclass": uncal_calib_metrics["brier_score_multiclass"],
            "negative_log_likelihood": uncal_calib_metrics["negative_log_likelihood"],
            "ece_top_label": uncal_calib_metrics["ece_top_label"],
            "mce_top_label": uncal_calib_metrics["mce_top_label"],
            "ece_positive_class": uncal_calib_metrics["ece_positive_class"],
            "mce_positive_class": uncal_calib_metrics["mce_positive_class"],
            "reliability_diagram_positive_class": uncal_calib_metrics["reliability_positive_class"],
            "reliability_diagram_top_label": uncal_calib_metrics["reliability_top_label"],
        },
        "test_metrics_calibrated": {
            "accuracy": cal_class_metrics["accuracy"],
            "precision_macro": cal_class_metrics["precision_macro"],
            "recall_macro": cal_class_metrics["recall_macro"],
            "f1_macro": cal_class_metrics["f1_macro"],
            "f1_binary": cal_class_metrics["f1_binary"],
            "roc_auc": cal_class_metrics["roc_auc"],
            "confusion_matrix": {
                "tp": cal_class_metrics["tp"],
                "fp": cal_class_metrics["fp"],
                "tn": cal_class_metrics["tn"],
                "fn": cal_class_metrics["fn"],
            },
            "brier_score_binary": cal_calib_metrics["brier_score_binary"],
            "brier_score_multiclass": cal_calib_metrics["brier_score_multiclass"],
            "negative_log_likelihood": cal_calib_metrics["negative_log_likelihood"],
            "ece_top_label": cal_calib_metrics["ece_top_label"],
            "mce_top_label": cal_calib_metrics["mce_top_label"],
            "ece_positive_class": cal_calib_metrics["ece_positive_class"],
            "mce_positive_class": cal_calib_metrics["mce_positive_class"],
            "reliability_diagram_positive_class": cal_calib_metrics["reliability_positive_class"],
            "reliability_diagram_top_label": cal_calib_metrics["reliability_top_label"],
        },
        "invariance_verification": {
            "discrete_predictions_identical": preds_identical,
            "accuracy_invariant": acc_identical,
            "roc_auc_invariant": auc_identical,
        },
        "metric_settings": {
            "num_bins": num_bins,
            "binning": "uniform_width_in_0_1",
            "score_definitions": {
                "brier_score_binary": "MSE on P(FAKE) in [0, 1]",
                "brier_score_multiclass": "Sum of MSE across both classes in [0, 2]",
                "negative_log_likelihood": "-mean(log(p_true)) with eps=1e-15",
                "ece_positive_class": "Marginal calibration of P(FAKE) over [0, 1] vs observed fake rate",
                "ece_top_label": "Confidence calibration of max_k P(k) over [0, 1] vs accuracy",
            },
        },
    }

    # 10. Write sanitized JSON output
    sanitized = sanitize_for_json(results_dict)
    with open(out_file_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    print(f"\n[Step 6/6] Calibration results safely written to:\n  {out_file_path.resolve()}")
    return sanitized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate probability calibration for STGNN on UPFD PolitiFact and GossipCop."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["politifact", "gossipcop", "all"],
        help="Dataset to evaluate: 'politifact', 'gossipcop', or 'all' (default: all)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/UPFD",
        help="Path to UPFD data directory (default: ./data/UPFD)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device (default: cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save JSON results (default: ./outputs)",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of bins for ECE and reliability diagrams (default: 10)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Progress log interval in cascades (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = ["politifact", "gossipcop"] if args.dataset == "all" else [args.dataset]

    overall_start = time.time()
    all_results = {}

    for name in datasets:
        ckpt_path = DEFAULT_CHECKPOINTS[name]
        res = evaluate_dataset_calibration(
            dataset_name=name,
            checkpoint_path=ckpt_path,
            data_root=args.data_root,
            device_str=args.device,
            output_dir=args.output_dir,
            num_bins=args.num_bins,
            log_interval=args.log_interval if name == "politifact" else max(args.log_interval, 200),
        )
        if res is not None:
            all_results[name] = res

    print("\n" + "=" * 75)
    print(f"  ALL CALIBRATION EVALUATIONS COMPLETED in {time.time() - overall_start:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
