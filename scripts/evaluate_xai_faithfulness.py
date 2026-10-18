"""
Broader Empirical XAI Faithfulness Evaluation for STGNN Misinformation Detection.

This script evaluates post-hoc node-feature and feature-modality attributions on
a deterministic sample of up to 10 held-out test cascades each from UPFD PolitiFact
and UPFD GossipCop.

Evaluation Protocol & Methodological Safeguards:
----------------------------------------------
1. Model & Data Integrity:
   - Uses official held-out test splits with sequential ordering (first 10 cascades: indices 0..9).
   - Reuses verified trained checkpoints without retraining or parameter fitting.
   - Enforces bit-for-bit prediction invariance before and after explanation passes.
2. Baselines for Faithfulness Comparison:
   - Attribution-Guided Progressive Masking: Nodes ranked by descending positive attribution (delta_p).
   - Random-Node Progressive Masking: Averaged over multiple fixed random seeds (default: 42, 43, 44, 45, 46).
   - Degree-Based Progressive Masking: Nodes ranked by descending out-degree in final cascade snapshot.
3. K-Value Range:
   - Targets practical deletion milestones: k in [1, 2, 5, 10], dynamically capped by available user nodes.
4. Sign Convention:
   - delta_p = baseline_probability - perturbed_probability.
   - Positive delta_p indicates a confidence decrease after masking (support for prediction).
   - Negative delta_p indicates a confidence increase (countervailing evidence).
5. Ground Truth & Error Analysis:
   - Tracks whether predictions were correct (TP/TN) vs incorrect (FP/FN) to evaluate faithfulness
     differences between confident correct predictions and error cases.
6. Non-Causal Scope:
   - Explanations measure sensitivity of the STGNN veracity classification to node features.
   - They do NOT identify real-world misinformation originators, deceivers, or patient zero.
   - The untrained STGNN auxiliary source_head is not used.

Output Destination:
   - outputs/xai_faithfulness_politifact.json
   - outputs/xai_faithfulness_gossipcop.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_upfd_data
from src.data.temporal_builder import TemporalCascadeBuilder
from src.models.stgnn import SpatioTemporalGNN
from src.evaluation.xai import (
    compute_modality_attribution,
    compute_node_attribution,
    _clone_snapshot_list,
    _run_forward,
)


# ---------------------------------------------------------------------
# Checkpoints & Constants
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

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46]
DEFAULT_TARGET_K: List[int] = [1, 2, 5, 10]
DEFAULT_SAMPLE_SIZE: int = 10


# ---------------------------------------------------------------------
# JSON Sanitization Helper
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
    if isinstance(val, np.integer):
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
# Model Loader
# ---------------------------------------------------------------------

def load_stgnn_checkpoint(
    checkpoint_path: Path,
    in_dim: int,
    device: torch.device,
) -> Tuple[SpatioTemporalGNN, Dict[str, Any]]:
    """Load pre-trained STGNN checkpoint in evaluation mode."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SpatioTemporalGNN(
        in_features=in_dim + 4,
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_snapshots=checkpoint.get("num_snapshots", 4),
        temporal_cell="gru",
        num_classes=2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "val_f1": checkpoint.get("val_f1"),
        "val_acc": checkpoint.get("val_acc"),
    }
    return model, meta


# ---------------------------------------------------------------------
# Subset Deletion Evaluator
# ---------------------------------------------------------------------

def _evaluate_subset_masking(
    model: torch.nn.Module,
    snapshots: List[Any],
    nodes_to_mask: Sequence[int],
    target_class: int,
    base_prob: float,
    baseline_value: float = 0.0,
) -> Tuple[float, float]:
    """
    Mask a specific subset of nodes across active snapshots and return (pert_prob, drop).
    """
    mask_set = set(nodes_to_mask)
    perturbed_snapshots = _clone_snapshot_list(snapshots)

    for snap in perturbed_snapshots:
        mask = getattr(snap, "active_mask", None)
        for nid in mask_set:
            is_active = True
            if mask is not None and nid < mask.numel():
                is_active = bool(mask.view(-1)[nid].item())
            if is_active:
                snap.x[nid, :] = baseline_value

    _, pert_probs = _run_forward(model, perturbed_snapshots)
    pert_prob = float(pert_probs[target_class].item())
    drop = base_prob - pert_prob
    return pert_prob, drop


# ---------------------------------------------------------------------
# Single Cascade Faithfulness Evaluator
# ---------------------------------------------------------------------

def evaluate_single_cascade(
    model: torch.nn.Module,
    cascade: Any,
    cascade_idx: int,
    temporal_builder: TemporalCascadeBuilder,
    target_k_values: Sequence[int] = DEFAULT_TARGET_K,
    random_seeds: Sequence[int] = DEFAULT_SEEDS,
    baseline_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluate feature-modality ablation, node attribution, and multi-seed faithfulness
    for a single cascade.
    """
    num_nodes = int(cascade.x.size(0))
    actual_class = int(cascade.y.item())
    actual_label = "FAKE" if actual_class == 1 else "REAL"

    # Build snapshots
    snapshots = temporal_builder.build_snapshots(cascade)

    # 1. Unperturbed Baseline Prediction
    base_logits_before, base_probs_before = _run_forward(model, snapshots)
    predicted_class = int(torch.argmax(base_probs_before).item())
    predicted_label = "FAKE" if predicted_class == 1 else "REAL"
    confidence = float(base_probs_before[predicted_class].item())

    is_correct = (predicted_class == actual_class)
    if is_correct:
        error_type = "true_positive" if actual_class == 1 else "true_negative"
    else:
        error_type = "false_positive" if predicted_class == 1 else "false_negative"

    # 2. Modality Attribution
    modality_res = compute_modality_attribution(
        model=model,
        snapshots=snapshots,
        target_class=predicted_class,
        baseline_value=baseline_value,
    )

    # 3. Node-Feature Attribution (Separating Root Node 0)
    node_res = compute_node_attribution(
        model=model,
        snapshots=snapshots,
        target_class=predicted_class,
        include_root=False,
        baseline_value=baseline_value,
    )

    root_attribution = node_res["root_attribution"]
    user_attributions = node_res["node_attributions"]

    # Collect eligible propagating user nodes
    eligible_user_nodes: List[int] = []
    user_deltas: Dict[int, float] = {}

    for nid, u_data in user_attributions.items():
        if nid == 0:
            continue
        eligible_user_nodes.append(nid)
        user_deltas[nid] = float(u_data["delta_probability"])

    n_eligible = len(eligible_user_nodes)

    # Valid k values capped by available user nodes
    valid_k_list = sorted(list({min(k, n_eligible) for k in target_k_values if k > 0 and n_eligible > 0}))

    # 4. Strategy Rankings
    # Strategy A: Attribution Ranking (descending positive delta_p)
    attr_ranked = sorted(eligible_user_nodes, key=lambda nid: user_deltas.get(nid, 0.0), reverse=True)

    # Strategy B: Degree Ranking (descending out-degree in final snapshot)
    final_snap = snapshots[-1]
    if final_snap.edge_index is not None and final_snap.edge_index.numel() > 0:
        src_nodes = final_snap.edge_index[0].detach().cpu()
        deg_counts = torch.bincount(src_nodes, minlength=num_nodes).numpy()
    else:
        deg_counts = np.zeros(num_nodes, dtype=np.int64)

    degree_ranked = sorted(
        eligible_user_nodes,
        key=lambda nid: (deg_counts[nid] if nid < len(deg_counts) else 0, user_deltas.get(nid, 0.0)),
        reverse=True,
    )

    # Strategy C: Random Ranking across multiple deterministic seeds
    random_rankings: Dict[int, List[int]] = {}
    for seed in random_seeds:
        rng = np.random.RandomState(seed)
        shuffled = list(eligible_user_nodes)
        rng.shuffle(shuffled)
        random_rankings[seed] = shuffled

    # 5. Evaluate Progressive Deletion Curves
    k_comparison_records: List[Dict[str, Any]] = []

    for k in valid_k_list:
        # Attribution drop
        attr_nodes = attr_ranked[:k]
        attr_pert_p, attr_drop = _evaluate_subset_masking(
            model, snapshots, attr_nodes, predicted_class, confidence, baseline_value
        )

        # Degree drop
        deg_nodes = degree_ranked[:k]
        deg_pert_p, deg_drop = _evaluate_subset_masking(
            model, snapshots, deg_nodes, predicted_class, confidence, baseline_value
        )

        # Random drops across seeds
        rand_drops: List[float] = []
        for seed, r_rank in random_rankings.items():
            r_nodes = r_rank[:k]
            _, r_drop = _evaluate_subset_masking(
                model, snapshots, r_nodes, predicted_class, confidence, baseline_value
            )
            rand_drops.append(r_drop)

        rand_mean_drop = float(np.mean(rand_drops))
        rand_std_drop = float(np.std(rand_drops))

        # Comparative outcomes
        attr_beats_random = bool(attr_drop > rand_mean_drop)
        attr_beats_degree = bool(attr_drop > deg_drop)
        attr_beats_both = bool(attr_beats_random and attr_beats_degree)

        k_comparison_records.append({
            "k": k,
            "attribution_masked_nodes": attr_nodes,
            "attribution_drop": attr_drop,
            "attribution_perturbed_p": attr_pert_p,
            "degree_masked_nodes": deg_nodes,
            "degree_drop": deg_drop,
            "degree_perturbed_p": deg_pert_p,
            "random_mean_drop": rand_mean_drop,
            "random_std_drop": rand_std_drop,
            "random_seed_drops": rand_drops,
            "attr_beats_random": attr_beats_random,
            "attr_beats_degree": attr_beats_degree,
            "attr_beats_both": attr_beats_both,
        })

    # 6. Post-Evaluation Invariance Verification
    base_logits_after, base_probs_after = _run_forward(model, snapshots)
    logits_diff = float(torch.max(torch.abs(base_logits_before - base_logits_after)).item())
    probs_diff = float(torch.max(torch.abs(base_probs_before - base_probs_after)).item())
    is_invariant = bool(logits_diff == 0.0 and probs_diff == 0.0)

    # Top-3 supporting and countervailing nodes
    sorted_users = sorted(user_attributions.values(), key=lambda x: x["delta_probability"], reverse=True)
    top_supporting = [
        {"node_id": x["node_id"], "delta_p": x["delta_probability"], "snaps": x["active_snapshots_count"]}
        for x in sorted_users if x["delta_probability"] > 0
    ][:3]
    top_countervailing = [
        {"node_id": x["node_id"], "delta_p": x["delta_probability"], "snaps": x["active_snapshots_count"]}
        for x in reversed(sorted_users) if x["delta_probability"] < 0
    ][:3]

    return {
        "cascade_id": cascade_idx,
        "num_nodes": num_nodes,
        "num_eligible_user_nodes": n_eligible,
        "actual_label": actual_label,
        "actual_class": actual_class,
        "predicted_label": predicted_label,
        "predicted_class": predicted_class,
        "is_correct": is_correct,
        "error_type": error_type,
        "baseline_confidence": confidence,
        "baseline_logits": [float(x) for x in base_logits_before.detach().cpu().tolist()],
        "baseline_probabilities": [float(x) for x in base_probs_before.detach().cpu().tolist()],
        "modality_attributions": modality_res["modality_attributions"],
        "root_attribution": root_attribution,
        "top_supporting_user_nodes": top_supporting,
        "top_countervailing_user_nodes": top_countervailing,
        "evaluated_k_values": valid_k_list,
        "k_comparisons": k_comparison_records,
        "invariance_verified": is_invariant,
        "max_logits_diff": logits_diff,
        "max_probs_diff": probs_diff,
    }


# ---------------------------------------------------------------------
# Aggregate Summary Statistics
# ---------------------------------------------------------------------

def _compute_stats_for_subset(records: List[Dict[str, Any]], target_k_values: Sequence[int]) -> Dict[str, Any]:
    """Compute mean, median, min, max, std, and win-rates for a subset of cascades."""
    if not records:
        return {"num_cascades": 0, "k_metrics": {}}

    k_metrics: Dict[str, Any] = {}

    for k in target_k_values:
        # Collect values for cascades that evaluated this k
        attr_drops: List[float] = []
        rand_drops: List[float] = []
        deg_drops: List[float] = []
        beats_rand_count = 0
        beats_deg_count = 0
        beats_both_count = 0

        for r in records:
            k_entry = next((entry for entry in r["k_comparisons"] if entry["k"] == k), None)
            if k_entry is not None:
                attr_drops.append(k_entry["attribution_drop"])
                rand_drops.append(k_entry["random_mean_drop"])
                deg_drops.append(k_entry["degree_drop"])
                if k_entry["attr_beats_random"]:
                    beats_rand_count += 1
                if k_entry["attr_beats_degree"]:
                    beats_deg_count += 1
                if k_entry["attr_beats_both"]:
                    beats_both_count += 1

        n_k = len(attr_drops)
        if n_k > 0:
            k_metrics[f"k={k}"] = {
                "evaluated_cascades_count": n_k,
                "attribution_drop": {
                    "mean": float(np.mean(attr_drops)),
                    "median": float(np.median(attr_drops)),
                    "std": float(np.std(attr_drops)),
                    "min": float(np.min(attr_drops)),
                    "max": float(np.max(attr_drops)),
                },
                "random_drop": {
                    "mean": float(np.mean(rand_drops)),
                    "median": float(np.median(rand_drops)),
                    "std": float(np.std(rand_drops)),
                    "min": float(np.min(rand_drops)),
                    "max": float(np.max(rand_drops)),
                },
                "degree_drop": {
                    "mean": float(np.mean(deg_drops)),
                    "median": float(np.median(deg_drops)),
                    "std": float(np.std(deg_drops)),
                    "min": float(np.min(deg_drops)),
                    "max": float(np.max(deg_drops)),
                },
                "comparative_outcomes": {
                    "attr_beats_random_count": beats_rand_count,
                    "attr_beats_random_rate": float(beats_rand_count / n_k),
                    "attr_beats_degree_count": beats_deg_count,
                    "attr_beats_degree_rate": float(beats_deg_count / n_k),
                    "attr_beats_both_count": beats_both_count,
                    "attr_beats_both_rate": float(beats_both_count / n_k),
                },
            }

    return {
        "num_cascades": len(records),
        "k_metrics": k_metrics,
    }


def aggregate_faithfulness_results(
    cascade_results: List[Dict[str, Any]],
    target_k_values: Sequence[int] = DEFAULT_TARGET_K,
) -> Dict[str, Any]:
    """
    Produce comprehensive aggregate statistics broken down by:
    - Overall (all sampled cascades)
    - Correct predictions vs. Incorrect predictions
    - True Positives, True Negatives, False Positives, False Negatives
    """
    correct_records = [r for r in cascade_results if r["is_correct"]]
    incorrect_records = [r for r in cascade_results if not r["is_correct"]]

    tp_records = [r for r in cascade_results if r["error_type"] == "true_positive"]
    tn_records = [r for r in cascade_results if r["error_type"] == "true_negative"]
    fp_records = [r for r in cascade_results if r["error_type"] == "false_positive"]
    fn_records = [r for r in cascade_results if r["error_type"] == "false_negative"]

    real_records = [r for r in cascade_results if r["actual_class"] == 0]
    fake_records = [r for r in cascade_results if r["actual_class"] == 1]

    return {
        "overall": _compute_stats_for_subset(cascade_results, target_k_values),
        "correct_predictions": _compute_stats_for_subset(correct_records, target_k_values),
        "incorrect_predictions": _compute_stats_for_subset(incorrect_records, target_k_values),
        "by_error_type": {
            "true_positive": _compute_stats_for_subset(tp_records, target_k_values),
            "true_negative": _compute_stats_for_subset(tn_records, target_k_values),
            "false_positive": _compute_stats_for_subset(fp_records, target_k_values),
            "false_negative": _compute_stats_for_subset(fn_records, target_k_values),
        },
        "by_actual_label": {
            "real": _compute_stats_for_subset(real_records, target_k_values),
            "fake": _compute_stats_for_subset(fake_records, target_k_values),
        },
    }


# ---------------------------------------------------------------------
# Dataset Evaluation Runner
# ---------------------------------------------------------------------

def evaluate_dataset_faithfulness(
    dataset_name: str,
    checkpoint_path: Path,
    num_cascades: int = DEFAULT_SAMPLE_SIZE,
    data_root: str = "./data/UPFD",
    device_str: str = "cpu",
    output_dir: str = "./outputs",
    overwrite: bool = False,
    random_seeds: Sequence[int] = DEFAULT_SEEDS,
    target_k_values: Sequence[int] = DEFAULT_TARGET_K,
) -> Dict[str, Any]:
    """
    Execute faithfulness evaluation on a deterministic sample of test cascades.
    """
    dataset_name = dataset_name.lower()
    device = torch.device(device_str)

    output_path = Path(output_dir) / f"xai_faithfulness_{dataset_name}.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            f"Use --overwrite to allow replacing."
        )

    print("\n" + "=" * 70)
    print(f"  XAI FAITHFULNESS EVALUATION: UPFD-{dataset_name.upper()}")
    print("=" * 70)
    print(f"Dataset root : {data_root}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Device       : {device}")
    print(f"Sample size  : {num_cascades} held-out test cascades (sequential 0..{num_cascades-1})")
    print(f"Random seeds : {list(random_seeds)}")
    print(f"Target k     : {list(target_k_values)}")

    # 1. Load Dataset
    print(f"\n1. Loading test split for UPFD-{dataset_name.capitalize()}...")
    train_ds, val_ds, test_ds, meta = load_upfd_data(
        root=data_root,
        name=dataset_name,
        feature="bert",
        multimodal=True,
    )
    total_test = len(test_ds)
    in_dim = int(meta["input_dim"])
    print(f"   Total test cascades available: {total_test} (input_dim: {in_dim})")

    # 2. Load Model
    print("2. Loading model checkpoint...")
    model, ckpt_meta = load_stgnn_checkpoint(
        checkpoint_path=checkpoint_path,
        in_dim=in_dim,
        device=device,
    )
    print(f"   Loaded checkpoint from epoch {ckpt_meta.get('epoch', 'N/A')} (val_f1: {ckpt_meta.get('val_f1', 0.0):.4f})")

    # 3. Setup Temporal Builder
    temporal_builder = TemporalCascadeBuilder(
        num_snapshots=4,
        add_temporal_features=True,
        require_complete_timestamps=True,
    )

    # 4. Evaluate Sampled Cascades
    sample_count = min(num_cascades, total_test)
    print(f"\n3. Evaluating {sample_count} test cascades sequentially...")

    cascade_results: List[Dict[str, Any]] = []
    start_time = time.time()

    for i in range(sample_count):
        cascade = test_ds[i]
        c_res = evaluate_single_cascade(
            model=model,
            cascade=cascade,
            cascade_idx=i,
            temporal_builder=temporal_builder,
            target_k_values=target_k_values,
            random_seeds=random_seeds,
        )
        cascade_results.append(c_res)

        status_str = "CORRECT" if c_res["is_correct"] else f"WRONG ({c_res['error_type'].upper()})"
        k1_entry = next((e for e in c_res["k_comparisons"] if e["k"] == 1), None)
        k1_str = f"k=1: attr={k1_entry['attribution_drop']:+.5f}, rand={k1_entry['random_mean_drop']:+.5f}" if k1_entry else "k=1: N/A"
        print(f"   [{i+1:2d}/{sample_count:2d}] Cascade {i:3d} (nodes={c_res['num_nodes']:3d}, "
              f"act={c_res['actual_label']}, pred={c_res['predicted_label']} "
              f"conf={c_res['baseline_confidence']:.3f}) -> {status_str:18s} | {k1_str}")

    elapsed = time.time() - start_time
    print(f"\nCompleted {sample_count} cascades in {elapsed:.2f}s ({elapsed/sample_count:.3f}s/cascade).")

    # 5. Invariance Verification
    all_invariant = all(r["invariance_verified"] for r in cascade_results)
    max_logits_diff = max(r["max_logits_diff"] for r in cascade_results)
    max_probs_diff = max(r["max_probs_diff"] for r in cascade_results)

    print(f"\nInvariance Verification:")
    print(f"   All cascades invariant: {all_invariant}")
    print(f"   Max logits diff       : {max_logits_diff:.2e}")
    print(f"   Max probs diff        : {max_probs_diff:.2e}")
    assert all_invariant, "Invariance check failed for one or more cascades!"

    # 6. Aggregate Statistics
    aggregate_stats = aggregate_faithfulness_results(cascade_results, target_k_values)

    # Classification breakdown
    num_correct = sum(1 for r in cascade_results if r["is_correct"])
    num_tp = sum(1 for r in cascade_results if r["error_type"] == "true_positive")
    num_tn = sum(1 for r in cascade_results if r["error_type"] == "true_negative")
    num_fp = sum(1 for r in cascade_results if r["error_type"] == "false_positive")
    num_fn = sum(1 for r in cascade_results if r["error_type"] == "false_negative")

    # Print Summary Table
    print("\n" + "-" * 70)
    print(f"  AGGREGATE FAITHFULNESS SUMMARY: UPFD-{dataset_name.upper()}")
    print("-" * 70)
    print(f"Sample Accuracy: {num_correct}/{sample_count} ({num_correct/sample_count*100:.1f}%) "
          f"[TP={num_tp}, TN={num_tn}, FP={num_fp}, FN={num_fn}]")
    print(f"{'K':>4s} | {'Attr Mean Drop':>15s} | {'Rand Mean Drop':>15s} | {'Deg Mean Drop':>15s} | {'Attr>Rand':>10s} | {'Attr>Deg':>10s}")
    print("-" * 75)

    overall_km = aggregate_stats["overall"]["k_metrics"]
    for k in target_k_values:
        k_key = f"k={k}"
        if k_key in overall_km:
            km = overall_km[k_key]
            a_mean = km["attribution_drop"]["mean"]
            r_mean = km["random_drop"]["mean"]
            d_mean = km["degree_drop"]["mean"]
            win_r = km["comparative_outcomes"]["attr_beats_random_rate"] * 100
            win_d = km["comparative_outcomes"]["attr_beats_degree_rate"] * 100
            print(f"{k:4d} | {a_mean:+15.6f} | {r_mean:+15.6f} | {d_mean:+15.6f} | {win_r:9.1f}% | {win_d:9.1f}%")

    # 7. Package and Write JSON
    payload = {
        "dataset": dataset_name,
        "model": "stgnn",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": ckpt_meta,
        "evaluation_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "evaluation_duration_seconds": elapsed,
        "protocol": {
            "num_cascades_sampled": sample_count,
            "selection_rule": f"Sequential first {sample_count} held-out test cascades with complete timestamps (indices 0..{sample_count-1})",
            "target_k_values": list(target_k_values),
            "random_seeds": list(random_seeds),
            "baseline_intervention": "Zero-masking active node features with fixed graph topology",
            "sign_convention": (
                "delta_p = baseline_probability - perturbed_probability. "
                "Positive values (>0) indicate confidence drop after masking (positive support)."
            ),
        },
        "invariance_verification": {
            "all_cascades_invariant": all_invariant,
            "max_logits_absolute_diff": max_logits_diff,
            "max_probs_absolute_diff": max_probs_diff,
        },
        "sample_classification_metrics": {
            "total_sampled": sample_count,
            "num_correct": num_correct,
            "num_incorrect": sample_count - num_correct,
            "accuracy": float(num_correct / sample_count),
            "true_positives": num_tp,
            "true_negatives": num_tn,
            "false_positives": num_fp,
            "false_negatives": num_fn,
        },
        "aggregate_faithfulness": aggregate_stats,
        "per_cascade_results": cascade_results,
        "methodological_guardrails": {
            "causality_disclaimer": (
                "Attribution scores quantify model prediction sensitivity under feature occlusion. "
                "They do NOT establish real-world causal responsibility or patient-zero identity."
            ),
            "untrained_head_disclaimer": "The STGNN auxiliary source_head was never trained and was not used in any calculation.",
            "sample_limitation": (
                f"Evaluated on a deterministic {sample_count}-cascade sample from the held-out test split. "
                "Serves as an exploratory empirical investigation rather than a definitive whole-benchmark claim."
            ),
        },
    }

    sanitized_payload = sanitize_for_json(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sanitized_payload, f, indent=2)

    print(f"\nSaved evaluation results to: {output_path}")
    return sanitized_payload


# ---------------------------------------------------------------------
# Command-Line Parser & Entry Point
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Broader empirical XAI faithfulness evaluation for STGNN."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["politifact", "gossipcop", "all"],
        help="Dataset to evaluate (default: all).",
    )
    parser.add_argument(
        "--num_cascades",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of test cascades to evaluate sequentially (default: 10).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/UPFD",
        help="Root directory of UPFD data (default: ./data/UPFD).",
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
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save JSON results (default: ./outputs).",
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
        help="Computation device (default: cpu).",
    )
    parser.add_argument(
        "--random_seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated random seeds for random baseline (default: 42,43,44,45,46).",
    )
    parser.add_argument(
        "--target_k",
        type=str,
        default="1,2,5,10",
        help="Comma-separated target k values for deletion curve (default: 1,2,5,10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seeds = [int(s.strip()) for s in args.random_seeds.split(",") if s.strip()]
    k_vals = [int(k.strip()) for k in args.target_k.split(",") if k.strip()]

    datasets_to_run = ["politifact", "gossipcop"] if args.dataset == "all" else [args.dataset]

    checkpoints = {
        "politifact": Path(args.checkpoint_politifact) if args.checkpoint_politifact else DEFAULT_CHECKPOINTS["politifact"],
        "gossipcop": Path(args.checkpoint_gossipcop) if args.checkpoint_gossipcop else DEFAULT_CHECKPOINTS["gossipcop"],
    }

    for ds in datasets_to_run:
        evaluate_dataset_faithfulness(
            dataset_name=ds,
            checkpoint_path=checkpoints[ds],
            num_cascades=args.num_cascades,
            data_root=args.data_root,
            device_str=args.device,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            random_seeds=seeds,
            target_k_values=k_vals,
        )


if __name__ == "__main__":
    main()
