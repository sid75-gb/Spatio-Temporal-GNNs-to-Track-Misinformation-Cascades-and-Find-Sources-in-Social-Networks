"""
Early-Stage Cascade Evaluation for STGNN Misinformation Detection.

Evaluates trained Spatio-Temporal Graph Neural Network (STGNN) checkpoints
at chronological observation milestones:
    - 25% chronological observation (prefix T=1)
    - 50% chronological observation (prefix T=2)
    - 75% chronological observation (prefix T=3)
    - 100% full cascade observation (prefix T=4)

Evaluates UPFD Politifact and UPFD Gossipcop separately using official held-out test splits.
Uses the same held-out test cascades across all four stages wherever timestamps permit.
Prevents future-event leakage by using strictly chronological quantile cutoffs,
zeroing inactive node features, filtering future edges, and feeding variable-length
snapshot prefixes with model.num_snapshots configured appropriately.

Includes runtime verification suite:
    1. Root node (node 0) active at every snapshot.
    2. Monotonically nested active-node masks across all snapshots (V_t <= V_{t+1}).
    3. Inactive node features strictly zeroed out (no future feature leakage).
    4. Active edges only connect active nodes (no future structural leakage).
    5. Exact numerical comparison at 100% against the corresponding trained checkpoint benchmark.

Results are saved to:
    - outputs/early_stage_politifact_stgnn.json
    - outputs/early_stage_gossipcop_stgnn.json
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
from src.evaluation.metrics import compute_classification_metrics
from src.models.stgnn import SpatioTemporalGNN


# ---------------------------------------------------------------------
# Checkpoint and Baseline Mappings
# ---------------------------------------------------------------------

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

STAGE_DEFINITIONS = [
    {"label": "25%", "fraction": 0.25, "prefix_k": 1},
    {"label": "50%", "fraction": 0.50, "prefix_k": 2},
    {"label": "75%", "fraction": 0.75, "prefix_k": 3},
    {"label": "100%", "fraction": 1.00, "prefix_k": 4},
]


# ---------------------------------------------------------------------
# JSON Serializer
# ---------------------------------------------------------------------

def sanitize_for_json(val: Any) -> Any:
    """Recursively converts non-finite floats, numpy, and torch objects to standard JSON."""
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


# ---------------------------------------------------------------------
# Model Loader & Variable-Length Prefix Verification
# ---------------------------------------------------------------------

def load_and_verify_stgnn_model(
    checkpoint_path: Path,
    in_dim: int,
    device: torch.device,
) -> Tuple[SpatioTemporalGNN, Dict[str, Any]]:
    """
    Load a trained SpatioTemporalGNN model checkpoint and configure it for
    variable-length prefix forward passes without rigid length assertions.
    Verifies that prefixes of lengths T in {1, 2, 3, 4} can be processed directly.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Please verify checkpoint directory."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    hidden_dim = int(checkpoint.get("hidden_dim", 64))

    # STGNN expects (in_dim + 4) input features when temporal features are attached
    # Setting num_snapshots=0 bypasses the fixed-T assertion, enabling
    # variable-length prefix sequences [1, 2, 3, 4] through dynamic attention & GRU.
    model = SpatioTemporalGNN(
        in_features=in_dim + 4,
        hidden_dim=hidden_dim,
        num_snapshots=0,
        temporal_cell="gru",
        num_classes=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def verify_variable_length_forward(
    model: SpatioTemporalGNN,
    sample_snapshots: List[Any],
    device: torch.device,
) -> None:
    """
    Directly tests model.forward_snapshots on prefixes of length 1, 2, 3, 4
    to verify that variable-length prefixes can be processed without blocker.
    """
    print("\n--- Verifying Variable-Length Prefix Forward Pass ---")
    for k in [1, 2, 3, 4]:
        prefix = [s.to(device) for s in sample_snapshots[:k]]
        with torch.no_grad():
            logits, _, _ = model.forward_snapshots(prefix)
            probs = F.softmax(logits, dim=-1)
        assert logits.shape == (1, 2), f"Expected logits shape (1, 2), got {logits.shape}"
        assert torch.isfinite(logits).all(), f"Non-finite logits detected at prefix length {k}"
        assert torch.isfinite(probs).all(), f"Non-finite probabilities detected at prefix length {k}"
        pred_label = "Fake" if torch.argmax(logits, dim=-1).item() == 1 else "Real"
        print(f"  [PASSED] Prefix T={k}: Logits={logits.cpu().tolist()[0]}, Prob={probs.cpu().tolist()[0]} -> {pred_label}")
    print("  Variable-length prefix execution confirmed safe and valid.")


# ---------------------------------------------------------------------
# Integrity Verification per Cascade
# ---------------------------------------------------------------------

def verify_cascade_snapshot_integrity(
    cascade_idx: int,
    snapshots: List[Any],
) -> None:
    """
    Verifies temporal snapshot invariants for a cascade:
    1. Root node (index 0) is active at every snapshot.
    2. Active-node masks are nested (never fewer active nodes at later snapshots).
    3. Inactive node features are zeroed out (no future feature leakage).
    4. Active edges only connect active nodes (no future edge leakage).
    """
    num_snaps = len(snapshots)

    # 1. Root node active
    for t in range(num_snaps):
        assert snapshots[t].active_mask[0].item() is True, (
            f"Cascade {cascade_idx}, snap {t}: Root node 0 is not active."
        )

    # 2. Monotonically nested active masks: active_mask[t] <= active_mask[t+1]
    for t in range(num_snaps - 1):
        mask_curr = snapshots[t].active_mask
        mask_next = snapshots[t + 1].active_mask
        # Logical implication: mask_curr -> mask_next <=> (~mask_curr | mask_next)
        nested = (~mask_curr | mask_next).all().item()
        assert nested, (
            f"Cascade {cascade_idx}: Active mask violation between snapshot {t} and {t+1}."
        )
        assert mask_curr.sum().item() <= mask_next.sum().item(), (
            f"Cascade {cascade_idx}: Active node count decreased from snap {t} to {t+1}."
        )

    # 3. Inactive node features zeroed
    for t in range(num_snaps):
        inactive_mask = ~snapshots[t].active_mask
        if inactive_mask.any():
            inactive_features = snapshots[t].x[inactive_mask]
            assert torch.all(inactive_features == 0.0), (
                f"Cascade {cascade_idx}, snap {t}: Inactive node features contain non-zero values."
            )

    # 4. Edges connect only active nodes
    for t in range(num_snaps):
        edge_index = snapshots[t].edge_index
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            mask = snapshots[t].active_mask
            valid_edges = (mask[src] & mask[dst]).all().item()
            assert valid_edges, (
                f"Cascade {cascade_idx}, snap {t}: Edges connect inactive/future nodes."
            )


# ---------------------------------------------------------------------
# Evaluation Core
# ---------------------------------------------------------------------

def evaluate_dataset_early_stage(
    dataset_name: str,
    checkpoint_path: Path,
    data_root: str = "./data/UPFD",
    device_str: str = "cpu",
    output_dir: str = "./outputs",
    log_interval: int = 100,
    baseline_path: Optional[Path] = None,
    output_filename: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a dataset across all early-stage milestones using identical held-out test cascades.
    """
    dataset_name = dataset_name.lower()
    device = torch.device(device_str)

    # Determine destination output path before performing expensive evaluation
    out_dir_path = Path(output_dir)
    out_filename = output_filename if output_filename else f"early_stage_{dataset_name}_stgnn.json"
    out_path = out_dir_path / out_filename

    if out_path.exists() and not overwrite:
        print("\n" + "=" * 70)
        print("  [ABORTED] Destination output file already exists:")
        print(f"    {out_path.resolve()}")
        print("  Strict safety constraint: Existing early-stage outputs cannot be overwritten.")
        print("  Use --overwrite to explicitly permit replacing existing files.")
        print("=" * 70 + "\n")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  EARLY-STAGE EVALUATION: UPFD-{dataset_name.upper()}")
    print("=" * 70)
    print(f"Dataset root : {data_root}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Device       : {device}")
    print(f"Stages       : {[s['label'] for s in STAGE_DEFINITIONS]}")
    print("Configuration: Variable-length prefix (model.num_snapshots=0)")

    # 1. Load dataset
    print(f"\nLoading test split for UPFD-{dataset_name.capitalize()}...")
    train_ds, val_ds, test_ds, meta = load_upfd_data(
        root=data_root,
        name=dataset_name,
        feature="bert",
        multimodal=True,
    )
    total_test_cascades = len(test_ds)
    in_dim = int(meta["input_dim"])
    print(f"Found {total_test_cascades} total test cascades (in_dim={in_dim}).")

    # 2. Load model
    print("Loading model checkpoint...")
    model, ckpt_meta = load_and_verify_stgnn_model(
        checkpoint_path=checkpoint_path,
        in_dim=in_dim,
        device=device,
    )
    print(
        f"Loaded checkpoint trained at epoch {ckpt_meta.get('epoch', 'N/A')} "
        f"(val_f1: {ckpt_meta.get('val_f1', 0.0):.4f})."
    )

    # 3. Setup temporal builder
    temporal_builder = TemporalCascadeBuilder(
        num_snapshots=4,
        add_temporal_features=True,
        require_complete_timestamps=True,
    )

    # 4. Verify variable-length forward pass on cascade 0
    sample_cascade = test_ds[0]
    sample_snaps = temporal_builder.build_snapshots(sample_cascade)
    verify_variable_length_forward(model, sample_snaps, device)

    # Track predictions per stage
    stage_data: Dict[str, Dict[str, List[Any]]] = {
        s["label"]: {"labels": [], "preds": [], "probs": []}
        for s in STAGE_DEFINITIONS
    }

    evaluated_cascades = 0
    skipped_cascades = 0
    skip_reasons: Dict[str, int] = {}
    verified_cascades = 0
    t_start = time.time()

    print(f"\nEvaluating {total_test_cascades} held-out test cascades across all 4 stages...")

    for idx in range(total_test_cascades):
        cascade = test_ds[idx]

        # Build 4-snapshot temporal progression strictly from chronological timestamps
        try:
            snapshots = temporal_builder.build_snapshots(cascade)
        except Exception as exc:
            skipped_cascades += 1
            reason = str(exc).split("\n")[0][:60]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        if len(snapshots) != 4:
            skipped_cascades += 1
            reason = f"Incomplete snapshot count ({len(snapshots)} != 4)"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        # Invariant verifications: root active, nested masks, zeroed inactive, valid edges
        verify_cascade_snapshot_integrity(idx, snapshots)
        verified_cascades += 1

        evaluated_cascades += 1
        y_true = int(cascade.y.item())

        # Evaluate all 4 observation stages on this identical cascade
        for stage in STAGE_DEFINITIONS:
            k = stage["prefix_k"]
            label = stage["label"]

            # Prefix slice: strictly earlier events, zero future leakage
            prefix_snapshots = [s.to(device) for s in snapshots[:k]]

            with torch.no_grad():
                logits, _, _ = model.forward_snapshots(prefix_snapshots)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = int(torch.argmax(logits, dim=-1).item())

            stage_data[label]["labels"].append(y_true)
            stage_data[label]["preds"].append(pred)
            stage_data[label]["probs"].append(probs)

        if (idx + 1) % log_interval == 0 or (idx + 1) == total_test_cascades:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_test_cascades - (idx + 1)) / rate if rate > 0 else 0
            print(
                f"  [{idx + 1:5d}/{total_test_cascades:5d}] "
                f"Evaluated: {evaluated_cascades:5d} | "
                f"Skipped: {skipped_cascades:3d} | "
                f"Speed: {rate:5.1f} casc/s | "
                f"ETA: {remaining:5.1f}s",
                flush=True,
            )

    total_time = time.time() - t_start
    print(f"\nCompleted evaluation of {evaluated_cascades}/{total_test_cascades} cascades in {total_time:.2f}s.")
    print(f"Structural and temporal integrity verified for all {verified_cascades} evaluated cascades.")
    if skipped_cascades > 0:
        print(f"Skipped {skipped_cascades} cascades. Reasons: {skip_reasons}")
    else:
        print("Skipped cascades: 0 (100% of held-out test cascades have complete official timestamp coverage).")

    # 5. Compute metrics for each stage
    stage_results: Dict[str, Dict[str, Any]] = {}
    table_rows = []

    for stage in STAGE_DEFINITIONS:
        label = stage["label"]
        fraction = stage["fraction"]
        prefix_k = stage["prefix_k"]

        labels = stage_data[label]["labels"]
        preds = stage_data[label]["preds"]
        probs_array = np.array(stage_data[label]["probs"])

        metrics = compute_classification_metrics(
            y_true=labels,
            y_pred=preds,
            y_probs=probs_array,
        )

        n_eval = len(labels)
        n_real = sum(1 for y in labels if y == 0)
        n_fake = sum(1 for y in labels if y == 1)

        stage_results[label] = {
            "stage": label,
            "observation_fraction": fraction,
            "prefix_snapshots": prefix_k,
            "num_cascades_evaluated": n_eval,
            "num_real": n_real,
            "num_fake": n_fake,
            "metrics": {
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_binary": metrics["f1_binary"],
                "roc_auc": metrics["roc_auc"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            },
        }

        table_rows.append([
            label,
            f"{fraction:.2f}",
            prefix_k,
            n_eval,
            f"{metrics['accuracy'] * 100:.2f}%",
            f"{metrics['f1_macro']:.4f}",
            f"{metrics['f1_binary']:.4f}",
            f"{metrics['precision_macro']:.4f}",
            f"{metrics['recall_macro']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            int(metrics["tp"]),
            int(metrics["fp"]),
            int(metrics["tn"]),
            int(metrics["fn"]),
        ])

    # 6. Display formatted summary table
    headers = [
        "Stage",
        "Obs Frac",
        "Prefix (T)",
        "N Eval",
        "Accuracy",
        "Macro F1",
        "Binary F1",
        "Macro Prec",
        "Macro Rec",
        "ROC-AUC",
        "TP",
        "FP",
        "TN",
        "FN",
    ]

    print("\n" + "=" * 80)
    print(f"  EARLY-STAGE SUMMARY: {dataset_name.upper()}")
    print("=" * 80)
    if HAS_TABULATE:
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))
    else:
        header_line = " | ".join(f"{h:>10}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        for row in table_rows:
            print(" | ".join(f"{str(v):>10}" for v in row))
    print("=" * 80)

    # 7. Exact comparison against baseline benchmark at 100%
    if baseline_path is None:
        baseline_path = BASELINE_RESULT_PATHS.get(dataset_name)
    baseline_match = False
    if baseline_path and Path(baseline_path).exists():
        baseline_path = Path(baseline_path)
        print(f"\n--- Benchmark Verification Check at 100% Observation ---")
        print(f"Comparing against baseline result file: {baseline_path.name}")
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        base_metrics = base_data.get("metrics", {})

        eval_100 = stage_results["100%"]["metrics"]
        diff_acc = abs(eval_100["accuracy"] - base_metrics.get("accuracy", 0.0))
        diff_f1 = abs(eval_100["f1_macro"] - base_metrics.get("f1_macro", 0.0))
        diff_auc = abs(eval_100["roc_auc"] - base_metrics.get("roc_auc", 0.0))
        cm_match = (
            eval_100["tp"] == base_metrics.get("tp")
            and eval_100["fp"] == base_metrics.get("fp")
            and eval_100["tn"] == base_metrics.get("tn")
            and eval_100["fn"] == base_metrics.get("fn")
        )

        print(f"  Baseline 100% Acc: {base_metrics.get('accuracy', 0.0):.6f} | Evaluated 100% Acc: {eval_100['accuracy']:.6f} (Diff: {diff_acc:.2e})")
        print(f"  Baseline 100% F1 : {base_metrics.get('f1_macro', 0.0):.6f} | Evaluated 100% F1 : {eval_100['f1_macro']:.6f} (Diff: {diff_f1:.2e})")
        print(f"  Baseline 100% AUC: {base_metrics.get('roc_auc', 0.0):.6f} | Evaluated 100% AUC: {eval_100['roc_auc']:.6f} (Diff: {diff_auc:.2e})")
        print(f"  Confusion Matrix Match (TP/FP/TN/FN): {cm_match} (TP={eval_100['tp']}, FP={eval_100['fp']}, TN={eval_100['tn']}, FN={eval_100['fn']})")

        if diff_acc < 1e-5 and diff_f1 < 1e-5 and cm_match:
            baseline_match = True
            print("  [VERIFIED] Exact numerical equivalence at 100% observation confirmed.")
        else:
            print("  [WARNING] Discrepancy found between 100% evaluation and baseline.")

    # 8. Save results JSON
    os.makedirs(output_dir, exist_ok=True)
    out_filename = output_filename if output_filename else f"early_stage_{dataset_name}_stgnn.json"
    out_path = Path(output_dir) / out_filename

    summary_json = {
        "dataset": dataset_name,
        "model": "stgnn",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "temporal_mode": "timestamp",
        "snapshot_configuration": "variable_length_prefix (model.num_snapshots=0)",
        "total_test_cascades": total_test_cascades,
        "evaluated_test_cascades": evaluated_cascades,
        "skipped_test_cascades": skipped_cascades,
        "skip_reasons": skip_reasons,
        "invariants_verified": {
            "root_active_all_phases": True,
            "nested_active_masks": True,
            "inactive_node_features_zeroed": True,
            "future_edges_excluded": True,
            "exact_100_percent_baseline_match": baseline_match,
        },
        "evaluation_duration_seconds": round(total_time, 2),
        "stages": stage_results,
        "summary_table": [
            {
                "stage": r[0],
                "observation_fraction": float(r[1]),
                "prefix_snapshots": r[2],
                "num_cascades_evaluated": r[3],
                "accuracy": stage_results[r[0]]["metrics"]["accuracy"],
                "f1_macro": stage_results[r[0]]["metrics"]["f1_macro"],
                "f1_binary": stage_results[r[0]]["metrics"]["f1_binary"],
                "precision_macro": stage_results[r[0]]["metrics"]["precision_macro"],
                "recall_macro": stage_results[r[0]]["metrics"]["recall_macro"],
                "roc_auc": stage_results[r[0]]["metrics"]["roc_auc"],
                "tp": stage_results[r[0]]["metrics"]["tp"],
                "fp": stage_results[r[0]]["metrics"]["fp"],
                "tn": stage_results[r[0]]["metrics"]["tn"],
                "fn": stage_results[r[0]]["metrics"]["fn"],
            }
            for r in table_rows
        ],
    }

    sanitized = sanitize_for_json(summary_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    print(f"\nSaved early-stage results to: {out_path.resolve()}")
    return sanitized


# ---------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate STGNN early-stage detection on UPFD Politifact and Gossipcop."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["politifact", "gossipcop", "all"],
        help="Dataset to evaluate: 'politifact', 'gossipcop', or 'all' (default: all)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Override checkpoint path for single dataset evaluation.",
    )
    parser.add_argument(
        "--checkpoint_politifact",
        type=str,
        default=None,
        help="Override checkpoint path for PolitiFact.",
    )
    parser.add_argument(
        "--checkpoint_gossipcop",
        type=str,
        default=None,
        help="Override checkpoint path for GossipCop.",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        default=None,
        help="Override baseline result path for 100% verification check.",
    )
    parser.add_argument(
        "--baseline_politifact",
        type=str,
        default=None,
        help="Override baseline result path for PolitiFact.",
    )
    parser.add_argument(
        "--baseline_gossipcop",
        type=str,
        default=None,
        help="Override baseline result path for GossipCop.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Custom output filename for single dataset (default: early_stage_{dataset}_stgnn.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output JSON files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device (default: cpu)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/UPFD",
        help="Path to UPFD data directory (default: ./data/UPFD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save JSON results (default: ./outputs)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log progress every N cascades (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = ["politifact", "gossipcop"] if args.dataset == "all" else [args.dataset]

    overall_start = time.time()
    results = {}

    checkpoints = {
        "politifact": Path(args.checkpoint_politifact or (args.checkpoint_path if args.dataset == "politifact" else None) or DEFAULT_CHECKPOINTS["politifact"]),
        "gossipcop": Path(args.checkpoint_gossipcop or (args.checkpoint_path if args.dataset == "gossipcop" else None) or DEFAULT_CHECKPOINTS["gossipcop"]),
    }

    baselines = {
        "politifact": Path(args.baseline_politifact or (args.baseline_path if args.dataset == "politifact" else None)) if (args.baseline_politifact or (args.baseline_path if args.dataset == "politifact" else None)) else BASELINE_RESULT_PATHS.get("politifact"),
        "gossipcop": Path(args.baseline_gossipcop or (args.baseline_path if args.dataset == "gossipcop" else None)) if (args.baseline_gossipcop or (args.baseline_path if args.dataset == "gossipcop" else None)) else BASELINE_RESULT_PATHS.get("gossipcop"),
    }

    if not args.overwrite:
        conflicts = []
        for name in datasets:
            out_fn = args.output_filename if len(datasets) == 1 else None
            fn = out_fn if out_fn else f"early_stage_{name}_stgnn.json"
            target_path = Path(args.output_dir) / fn
            if target_path.exists():
                conflicts.append(str(target_path.resolve()))
        if conflicts:
            print("\n" + "=" * 70)
            print("  [ABORTED] Destination output file(s) already exist:")
            for c in conflicts:
                print(f"    - {c}")
            print("  Strict safety constraint: Existing early-stage outputs cannot be overwritten.")
            print("  Use --overwrite to explicitly permit replacing existing files.")
            print("=" * 70 + "\n")
            sys.exit(1)

    for name in datasets:
        ckpt_path = checkpoints[name]
        b_path = baselines.get(name)
        out_fn = args.output_filename if len(datasets) == 1 else None
        res = evaluate_dataset_early_stage(
            dataset_name=name,
            checkpoint_path=ckpt_path,
            data_root=args.data_root,
            device_str=args.device,
            output_dir=args.output_dir,
            log_interval=args.log_interval if name == "politifact" else max(args.log_interval, 200),
            baseline_path=b_path,
            output_filename=out_fn,
            overwrite=args.overwrite,
        )
        if res is not None:
            results[name] = res

    print("\n" + "=" * 80)
    print(f"  ALL EARLY-STAGE EVALUATIONS COMPLETED in {time.time() - overall_start:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
