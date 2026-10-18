"""
Explainable AI (XAI) Evaluation Utilities for Spatio-Temporal GNNs.

This module provides post-hoc explanation methods for pre-trained, frozen
Spatio-Temporal Graph Neural Networks (STGNN) performing misinformation cascade
veracity classification.

Architecture & Feature Context:
------------------------------
1. Model:
   The STGNN (SpatioTemporalGNN in src/models/stgnn.py) processes temporal
   cascade snapshots [S_0, S_1, ..., S_{T-1}] using bidirectional GAT spatial
   diffusion, recurrent temporal cells (GRU/LSTM), temporal self-attention,
   global attention pooling, and an explicit root-node concatenation pathway:
       cascade_summary = concat(graph_pooled, root_embeds)
       veracity_logits = veracity_classifier(cascade_summary)

2. Input Feature Layout (UPFD Multimodal Benchmark):
   - Dimensions 0..767: News / tweet text BERT dense semantic embeddings (768 dims).
   - Dimensions 768..777: User Twitter profile features (10 dims, normalized via
     training-set statistics: follower count, friend count, verified status, etc.).
   - Dimensions 778..781: Engineered temporal/structural dynamics (4 dims:
     normalized depth, normalized out-degree, elapsed fraction, active indicator).
   Total input feature dimension = 782.

3. Explanation Methods Implemented:
   - Feature-Modality Ablation (compute_modality_attribution):
     Measures sensitivity of graph-level classification to macro semantic blocks
     (Content vs. Profile vs. Temporal Dynamics).
   - Node-Feature Occlusion (compute_node_attribution):
     Measures marginal change in target-class confidence when a specific node's
     feature representation is masked across its active snapshots.
   - Faithfulness Evaluation (evaluate_explanation_faithfulness):
     Progressively masks highest-attributed nodes and compares confidence
     deletion curves against random and structural degree-based baselines.

Scientific Safeguards & Methodological Scope:
--------------------------------------------
- Attribution vs. Causality:
  Attribution scores measure mathematical sensitivity of the classifier's
  output to input perturbations. They do NOT constitute causal proof of real-world
  originator, deceiver, or patient-zero status.
- Untrained Source Head:
  The auxiliary node-level `source_head` in STGNN was never trained or validated
  against ground-truth source labels. All explanations here pertain strictly to the
  trained graph-level veracity classification.
- Root Node Separation:
  Node 0 represents the published news claim/article, which directly occupies 50%
  of the classifier's summary vector (`root_embeds`). Root attribution is handled
  and reported separately from propagating user retweeter nodes.
- Topological Invariance:
  Perturbations operate strictly via feature masking. Graph topology (`edge_index`),
  node indices, and active masks remain structurally valid and unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


# ---------------------------------------------------------------------
# Verified Feature Modality Layout
# ---------------------------------------------------------------------

DEFAULT_FEATURE_GROUPS: Dict[str, Tuple[int, int]] = {
    "bert_content": (0, 768),
    "user_profile": (768, 778),
    "temporal_dynamics": (778, 782),
}

FEATURE_GROUP_DESCRIPTIONS: Dict[str, str] = {
    "bert_content": "Textual semantic representation of news claim or user tweet history (dims 0..767)",
    "user_profile": "Twitter account metadata normalized by training-set statistics (dims 768..777)",
    "temporal_dynamics": "Snapshot structural-temporal metrics: depth, degree, elapsed time, active flag (dims 778..781)",
}


# ---------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------

def _clone_snapshot_list(snapshot_list: List[Data]) -> List[Data]:
    """
    Perform a safe clone of a list of PyG Data snapshots.

    Guarantees that `x`, `edge_index`, and `active_mask` tensors are detached
    and cloned so that downstream perturbations never mutate the original data.
    """
    cloned_list: List[Data] = []
    for snap in snapshot_list:
        cloned = snap.clone()
        if cloned.x is not None:
            cloned.x = cloned.x.clone()
        if cloned.edge_index is not None:
            cloned.edge_index = cloned.edge_index.clone()
        if hasattr(cloned, "active_mask") and cloned.active_mask is not None:
            cloned.active_mask = cloned.active_mask.clone()
        cloned_list.append(cloned)
    return cloned_list


def _validate_snapshots(snapshot_list: List[Data]) -> Tuple[int, int, torch.device]:
    """
    Validate snapshot list non-emptiness, node count, and feature dimensions.

    Returns:
        num_nodes: Total number of nodes in the cascade.
        in_features: Number of feature columns per node.
        device: Device of the snapshot tensors.
    """
    if not snapshot_list or len(snapshot_list) == 0:
        raise ValueError("snapshot_list must contain at least one snapshot.")

    first_snap = snapshot_list[0]
    if first_snap.x is None or first_snap.x.numel() == 0:
        raise ValueError("First snapshot contains no node features (x is None or empty).")

    num_nodes = int(first_snap.x.size(0))
    in_features = int(first_snap.x.size(1))
    device = first_snap.x.device

    for idx, snap in enumerate(snapshot_list):
        if snap.x is None:
            raise ValueError(f"Snapshot {idx} has x=None.")
        if snap.x.size(0) != num_nodes:
            raise ValueError(
                f"Node count mismatch: snapshot 0 has {num_nodes} nodes, "
                f"but snapshot {idx} has {snap.x.size(0)} nodes."
            )
        if snap.x.size(1) != in_features:
            raise ValueError(
                f"Feature dimension mismatch: snapshot 0 has {in_features} features, "
                f"but snapshot {idx} has {snap.x.size(1)} features."
            )

    return num_nodes, in_features, device


@torch.no_grad()
def _run_forward(model: torch.nn.Module, snapshot_list: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run forward pass on a list of snapshots in evaluation mode.

    Returns:
        logits: 1D or 2D tensor of shape [2] representing raw class logits.
        probabilities: 1D tensor of shape [2] representing softmax probabilities [P(REAL), P(FAKE)].
    """
    was_training = model.training
    model.eval()
    try:
        # Move snapshots to model device if necessary
        model_device = next(model.parameters()).device
        device_snapshots = [
            snap.to(model_device) if snap.x.device != model_device else snap
            for snap in snapshot_list
        ]

        if hasattr(model, "forward_snapshots"):
            out = model.forward_snapshots(device_snapshots)
        else:
            out = model(device_snapshots)

        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        # Handle shape [1, 2] -> [2]
        if logits.dim() == 2 and logits.size(0) == 1:
            logits = logits.squeeze(0)

        probs = F.softmax(logits, dim=-1)
        return logits, probs
    finally:
        if was_training:
            model.train()


# ---------------------------------------------------------------------
# 1. Feature-Modality Ablation
# ---------------------------------------------------------------------

def compute_modality_attribution(
    model: torch.nn.Module,
    snapshots: List[Data],
    target_class: Optional[int] = None,
    feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
    baseline_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute sensitivity of the STGNN veracity prediction to feature modalities.

    Evaluates the marginal drop in target-class probability when each high-level
    feature modality is masked across all active snapshots.

    Sign Convention:
        delta_probability = baseline_probability - perturbed_probability
        - delta_probability > 0: Masking this modality reduces target-class confidence
          (the modality provided positive support for the target prediction).
        - delta_probability < 0: Masking this modality increases target-class confidence
          (the modality provided countervailing evidence against the target prediction).
        - delta_probability ~ 0: Target-class confidence is insensitive to this modality.

    Methodological Interpretation:
        These values represent non-linear ablation sensitivity under the configured
        baseline intervention. They must NOT be normalized into percentages or
        interpreted as additive causal importance weights.

    Parameters:
        model: Trained SpatioTemporalGNN model (evaluated with frozen weights).
        snapshots: List of chronological PyG Data snapshots for one cascade.
        target_class: Class index to explain (0 = REAL, 1 = FAKE). If None, defaults
            to the model's unperturbed predicted class: argmax(P(class)).
        feature_groups: Mapping of group names to (start_col, end_col) slices.
            Defaults to DEFAULT_FEATURE_GROUPS (bert_content, user_profile, temporal_dynamics).
        baseline_value: Scalar replacement value for masked features (default 0.0).

    Returns:
        Dictionary containing baseline prediction, target class, per-modality
        probability deltas, raw perturbed probabilities, and metadata.
    """
    num_nodes, in_features, _ = _validate_snapshots(snapshots)

    # Resolve feature groups
    if feature_groups is None:
        if in_features >= 782:
            groups = DEFAULT_FEATURE_GROUPS
        else:
            raise ValueError(
                f"Snapshot input feature dimension is {in_features}, but default UPFD "
                f"layout requires at least 782 features (768 BERT + 10 Profile + 4 Temporal). "
                f"Provide custom `feature_groups` for non-standard feature dimensions."
            )
    else:
        groups = feature_groups
        # Validate custom group boundaries
        for g_name, (start, end) in groups.items():
            if start < 0 or end > in_features or start >= end:
                raise ValueError(
                    f"Feature group '{g_name}' has invalid slice ({start}, {end}) "
                    f"for feature dimension {in_features}."
                )

    # 1. Establish unperturbed baseline prediction
    base_logits, base_probs = _run_forward(model, snapshots)

    predicted_class = int(torch.argmax(base_probs).item())
    if target_class is None:
        target_class = predicted_class
    else:
        target_class = int(target_class)
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 (REAL) or 1 (FAKE), got {target_class}.")

    base_prob = float(base_probs[target_class].item())

    # 2. Perform leave-one-group-out ablation
    modality_results: Dict[str, Dict[str, Any]] = {}

    for group_name, (start_col, end_col) in groups.items():
        # Clone snapshots so original data is strictly untouched
        perturbed_snapshots = _clone_snapshot_list(snapshots)

        for snap in perturbed_snapshots:
            # Mask the feature slice for all active nodes (and inactive)
            # Inactive nodes already have x=0; masking preserves that property.
            snap.x[:, start_col:end_col] = baseline_value

        # Recompute prediction
        pert_logits, pert_probs = _run_forward(model, perturbed_snapshots)
        pert_prob = float(pert_probs[target_class].item())
        delta_p = base_prob - pert_prob

        modality_results[group_name] = {
            "delta_probability": delta_p,
            "perturbed_probability": pert_prob,
            "feature_slice": [start_col, end_col],
            "description": FEATURE_GROUP_DESCRIPTIONS.get(group_name, f"Columns {start_col}..{end_col}"),
        }

    return {
        "target_class": target_class,
        "target_label": "FAKE" if target_class == 1 else "REAL",
        "predicted_class": predicted_class,
        "predicted_label": "FAKE" if predicted_class == 1 else "REAL",
        "baseline_probability": base_prob,
        "baseline_logits": [float(x) for x in base_logits.detach().cpu().tolist()],
        "modality_attributions": modality_results,
        "sign_convention": (
            "delta_probability = baseline_probability - perturbed_probability. "
            "Positive values (>0) indicate that masking this modality decreased confidence "
            "in the target class (positive support)."
        ),
        "baseline_intervention": f"Constant scalar substitution (value={baseline_value})",
        "score_interpretation": "Ablation sensitivity (non-additive marginal probability shift, not causal importance)",
    }


# ---------------------------------------------------------------------
# 2. Node-Feature Occlusion
# ---------------------------------------------------------------------

def compute_node_attribution(
    model: torch.nn.Module,
    snapshots: List[Data],
    target_class: Optional[int] = None,
    include_root: bool = False,
    baseline_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute sensitivity of the STGNN veracity prediction to individual node features.

    For each eligible active node, its feature representation is replaced by a
    baseline value across all snapshots where it is active. Crucially, the graph
    topology (`edge_index`), node indexing, and active masks remain untouched.

    Root Node Separation:
        In the UPFD benchmark, Node 0 represents the root news claim/article,
        not a propagating human retweeter. Furthermore, the STGNN architecture
        explicitly concatenates `root_embeds` into `cascade_summary`, meaning
        Node 0 has an architecturally dominant influence. To prevent misleading
        conflation with user retweeters, Node 0 attribution is stored in a separate
        `root_attribution` field and excluded from `node_attributions` unless
        explicitly requested.

    Sign Convention:
        delta_probability = baseline_probability - perturbed_probability
        - delta_probability > 0: Node features supported the target class prediction.
        - delta_probability < 0: Node features provided countervailing evidence.
        - delta_probability ~ 0: Node features were uninformative to the decision.

    Caveat on Feature Masking:
        Feature masking measures sensitivity to the node's feature content under
        fixed connectivity. It does NOT simulate complete topological deletion of
        the node or severance of its downstream retweet tree. In continuous BERT
        space, zero-vector substitution represents the absence of signal rather than
        a linguistically valid neutral text embedding.

    Parameters:
        model: Trained SpatioTemporalGNN model.
        snapshots: Chronological PyG Data snapshots for one cascade.
        target_class: Class to explain (0 = REAL, 1 = FAKE). Defaults to predicted class.
        include_root: If True, Node 0 is also included in `node_attributions`.
            In either case, Node 0 attribution is always provided in `root_attribution`.
        baseline_value: Scalar replacement value for masked features (default 0.0).

    Returns:
        Dictionary containing baseline prediction, `root_attribution`, and
        `node_attributions` for all active user nodes.
    """
    num_nodes, _, _ = _validate_snapshots(snapshots)

    # 1. Unperturbed baseline prediction
    base_logits, base_probs = _run_forward(model, snapshots)

    predicted_class = int(torch.argmax(base_probs).item())
    if target_class is None:
        target_class = predicted_class
    else:
        target_class = int(target_class)
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 (REAL) or 1 (FAKE), got {target_class}.")

    base_prob = float(base_probs[target_class].item())

    # 2. Identify active snapshots for each node
    # A node is active at snapshot t if active_mask[u] is True
    node_active_counts: Dict[int, int] = {node_id: 0 for node_id in range(num_nodes)}
    for snap in snapshots:
        mask = getattr(snap, "active_mask", None)
        if mask is not None:
            active_indices = torch.where(mask.view(-1))[0].tolist()
            for u in active_indices:
                if u in node_active_counts:
                    node_active_counts[u] += 1
        else:
            # If no active_mask, assume all nodes active
            for u in range(num_nodes):
                node_active_counts[u] += 1

    # 3. Helper to perturb a single node across active snapshots
    def _evaluate_single_node(node_id: int) -> Dict[str, Any]:
        perturbed_snapshots = _clone_snapshot_list(snapshots)
        active_snaps = 0

        for snap in perturbed_snapshots:
            mask = getattr(snap, "active_mask", None)
            is_active = True
            if mask is not None and node_id < mask.numel():
                is_active = bool(mask.view(-1)[node_id].item())

            if is_active:
                active_snaps += 1
                # Mask node features
                snap.x[node_id, :] = baseline_value

        # Forward pass on perturbed graph
        _, pert_probs = _run_forward(model, perturbed_snapshots)
        pert_prob = float(pert_probs[target_class].item())
        delta_p = base_prob - pert_prob

        return {
            "node_id": node_id,
            "delta_probability": delta_p,
            "perturbed_probability": pert_prob,
            "active_snapshots_count": active_snaps,
        }

    # 4. Compute Root Attribution (Node 0)
    root_attribution: Optional[Dict[str, Any]] = None
    if num_nodes > 0:
        root_res = _evaluate_single_node(0)
        root_attribution = {
            **root_res,
            "role": "Root News Item / Published Claim",
            "note": "Node 0 directly supplies root_embeds (50% of classifier input) in STGNN.",
        }

    # 5. Compute User / Retweeter Node Attributions (Nodes 1..N-1)
    user_attributions: Dict[int, Dict[str, Any]] = {}
    for node_id in range(1, num_nodes):
        if node_active_counts[node_id] > 0:
            res = _evaluate_single_node(node_id)
            user_attributions[node_id] = {
                **res,
                "role": "Propagating User / Retweeter",
            }

    # If include_root is explicitly requested, also copy root into user_attributions
    if include_root and root_attribution is not None:
        user_attributions[0] = root_attribution

    return {
        "target_class": target_class,
        "target_label": "FAKE" if target_class == 1 else "REAL",
        "predicted_class": predicted_class,
        "predicted_label": "FAKE" if predicted_class == 1 else "REAL",
        "baseline_probability": base_prob,
        "baseline_logits": [float(x) for x in base_logits.detach().cpu().tolist()],
        "root_attribution": root_attribution,
        "node_attributions": user_attributions,
        "total_active_user_nodes": len(user_attributions) - (1 if include_root and 0 in user_attributions else 0),
        "sign_convention": (
            "delta_probability = baseline_probability - perturbed_probability. "
            "Positive values (>0) indicate that masking this node decreased confidence "
            "in the target class (positive support)."
        ),
        "baseline_value": baseline_value,
        "disclaimer": (
            "Feature occlusion measures sensitivity to node features with fixed topology. "
            "It does not reflect edge removal, structural deletion, or real-world source causality."
        ),
    }


# ---------------------------------------------------------------------
# 3. Faithfulness Evaluation
# ---------------------------------------------------------------------

def evaluate_explanation_faithfulness(
    model: torch.nn.Module,
    snapshots: List[Data],
    target_class: Optional[int] = None,
    node_attributions: Optional[Dict[str, Any]] = None,
    top_k_list: Sequence[int] = (1, 2, 5, 10),
    baseline_value: float = 0.0,
    seed: int = 42,
    include_root_in_ranking: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate the empirical faithfulness of node-feature attributions.

    Progressively masks the highest-scoring node features and compares the resulting
    target-class confidence drop against two standard reference baselines:
    1. Random Node Deletion: Nodes selected uniformly at random without replacement.
    2. Degree-Based Deletion: Nodes selected by descending out-degree (structural hub baseline).

    Sign & Metric Definitions:
        confidence_drop(k) = baseline_probability - perturbed_probability(top_k_masked)
        - Higher confidence_drop indicates stronger impact when the subset is removed
          (Fidelity- / Comprehensiveness).
        - If attribution is faithful to the model's learned representation, attribution-guided
          deletion should cause a more rapid confidence drop than random or purely
          structural degree-based deletion.

    Safeguards:
        - Deterministic: Uses `np.random.RandomState(seed)` for reproducible random selections.
        - Adaptive K: Practical top-k values are capped by the number of available active nodes.
        - Non-judgmental: Reports raw deletion curves without imposing arbitrary pass/fail
          thresholds or claiming that execution alone proves faithfulness.

    Parameters:
        model: Trained STGNN model.
        snapshots: Chronological PyG Data snapshots for one cascade.
        target_class: Target class index to evaluate. Defaults to predicted class.
        node_attributions: Precomputed output of `compute_node_attribution`. If None,
            computed automatically.
        top_k_list: Sequence of k values for progressive masking (e.g., 1, 2, 5, 10).
        baseline_value: Scalar substitution value for masked node features.
        seed: Random seed for deterministic random-baseline selection.
        include_root_in_ranking: Whether root node 0 is eligible for progressive masking.
            Defaults to False (evaluates user propagation nodes strictly).

    Returns:
        Structured dictionary containing deletion curves for attribution, random,
        and degree baselines across all evaluated k values.
    """
    num_nodes, _, _ = _validate_snapshots(snapshots)

    # 1. Unperturbed baseline prediction
    base_logits, base_probs = _run_forward(model, snapshots)

    predicted_class = int(torch.argmax(base_probs).item())
    if target_class is None:
        target_class = predicted_class
    else:
        target_class = int(target_class)
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 (REAL) or 1 (FAKE), got {target_class}.")

    base_prob = float(base_probs[target_class].item())

    # 2. Obtain node attributions if not provided
    if node_attributions is None:
        node_attr_res = compute_node_attribution(
            model=model,
            snapshots=snapshots,
            target_class=target_class,
            include_root=include_root_in_ranking,
            baseline_value=baseline_value,
        )
    else:
        node_attr_res = node_attributions

    # Collect candidate user nodes
    user_attrs = node_attr_res.get("node_attributions", {})
    eligible_nodes: List[int] = []
    node_deltas: Dict[int, float] = {}

    for nid_key, val_dict in user_attrs.items():
        nid = int(nid_key)
        if not include_root_in_ranking and nid == 0:
            continue
        eligible_nodes.append(nid)
        node_deltas[nid] = float(val_dict["delta_probability"])

    # If root is included and not in user_attrs, add from root_attribution
    if include_root_in_ranking and 0 not in eligible_nodes:
        root_data = node_attr_res.get("root_attribution")
        if root_data is not None:
            eligible_nodes.append(0)
            node_deltas[0] = float(root_data["delta_probability"])

    n_eligible = len(eligible_nodes)

    # Cap top-k values by available eligible nodes
    valid_k_list = sorted(list({min(k, n_eligible) for k in top_k_list if k > 0 and n_eligible > 0}))

    if n_eligible == 0:
        return {
            "target_class": target_class,
            "target_label": "FAKE" if target_class == 1 else "REAL",
            "baseline_probability": base_prob,
            "evaluated_k_values": [],
            "total_eligible_nodes": 0,
            "attribution_deletion_curve": [],
            "random_deletion_curve": [],
            "degree_deletion_curve": [],
            "seed": seed,
            "note": "No eligible nodes available for progressive masking.",
        }

    # 3. Strategy Rankings
    # A. Attribution ranking: descending order of delta_probability (highest positive support first)
    attr_ranked = sorted(eligible_nodes, key=lambda nid: node_deltas.get(nid, 0.0), reverse=True)

    # B. Degree ranking: descending order of out-degree in the final snapshot
    final_snap = snapshots[-1]
    if final_snap.edge_index is not None and final_snap.edge_index.numel() > 0:
        src_nodes = final_snap.edge_index[0].detach().cpu()
        deg_counts = torch.bincount(src_nodes, minlength=num_nodes).numpy()
    else:
        deg_counts = np.zeros(num_nodes, dtype=np.int64)

    degree_ranked = sorted(
        eligible_nodes,
        key=lambda nid: (deg_counts[nid] if nid < len(deg_counts) else 0, node_deltas.get(nid, 0.0)),
        reverse=True,
    )

    # C. Random ranking: deterministic permutation
    rng = np.random.RandomState(seed)
    random_ranked = list(eligible_nodes)
    rng.shuffle(random_ranked)

    # 4. Helper to evaluate joint masking of a node subset
    def _evaluate_subset_masking(nodes_to_mask: Sequence[int]) -> Tuple[float, float]:
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

    # 5. Compute deletion curves
    attr_curve: List[Dict[str, Any]] = []
    rand_curve: List[Dict[str, Any]] = []
    deg_curve: List[Dict[str, Any]] = []

    for k in valid_k_list:
        # Attribution
        attr_nodes = attr_ranked[:k]
        attr_prob, attr_drop = _evaluate_subset_masking(attr_nodes)
        attr_curve.append({
            "k": k,
            "masked_nodes": attr_nodes,
            "perturbed_probability": attr_prob,
            "confidence_drop": attr_drop,
        })

        # Random
        rand_nodes = random_ranked[:k]
        rand_prob, rand_drop = _evaluate_subset_masking(rand_nodes)
        rand_curve.append({
            "k": k,
            "masked_nodes": rand_nodes,
            "perturbed_probability": rand_prob,
            "confidence_drop": rand_drop,
        })

        # Degree
        deg_nodes = degree_ranked[:k]
        deg_prob, deg_drop = _evaluate_subset_masking(deg_nodes)
        deg_curve.append({
            "k": k,
            "masked_nodes": deg_nodes,
            "perturbed_probability": deg_prob,
            "confidence_drop": deg_drop,
        })

    return {
        "target_class": target_class,
        "target_label": "FAKE" if target_class == 1 else "REAL",
        "baseline_probability": base_prob,
        "evaluated_k_values": valid_k_list,
        "total_eligible_nodes": n_eligible,
        "attribution_deletion_curve": attr_curve,
        "random_deletion_curve": rand_curve,
        "degree_deletion_curve": deg_curve,
        "seed": seed,
        "disclaimer": (
            "Fidelity curves report raw confidence drops under progressive feature masking. "
            "Higher drops indicate stronger sensitivity to the masked subset (Fidelity-). "
            "Relative performance between curves must be assessed empirically without arbitrary thresholding."
        ),
    }


# ---------------------------------------------------------------------
# 4. Comprehensive Cascade Explanation Wrapper
# ---------------------------------------------------------------------

def compute_cascade_explanations(
    model: torch.nn.Module,
    snapshots: List[Data],
    target_class: Optional[int] = None,
    feature_groups: Optional[Dict[str, Tuple[int, int]]] = None,
    include_root_in_ranking: bool = False,
    baseline_value: float = 0.0,
    evaluate_faithfulness: bool = False,
    faithfulness_k_list: Sequence[int] = (1, 2, 5, 10),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Convenience wrapper that executes modality ablation, node attribution,
    and optional faithfulness evaluation for a single cascade.

    Returns:
        Structured dictionary ready for serialization into dashboard JSON or
        research experiment records.
    """
    modality_res = compute_modality_attribution(
        model=model,
        snapshots=snapshots,
        target_class=target_class,
        feature_groups=feature_groups,
        baseline_value=baseline_value,
    )

    resolved_target = modality_res["target_class"]

    node_res = compute_node_attribution(
        model=model,
        snapshots=snapshots,
        target_class=resolved_target,
        include_root=include_root_in_ranking,
        baseline_value=baseline_value,
    )

    faithfulness_res: Optional[Dict[str, Any]] = None
    if evaluate_faithfulness:
        faithfulness_res = evaluate_explanation_faithfulness(
            model=model,
            snapshots=snapshots,
            target_class=resolved_target,
            node_attributions=node_res,
            top_k_list=faithfulness_k_list,
            baseline_value=baseline_value,
            seed=seed,
            include_root_in_ranking=include_root_in_ranking,
        )

    # Rank user nodes by descending attribution (positive drivers)
    user_attrs = node_res["node_attributions"]
    ranked_nodes = sorted(
        user_attrs.values(),
        key=lambda item: item["delta_probability"],
        reverse=True,
    )

    top_supporting_nodes = [
        {"node_id": item["node_id"], "delta_probability": item["delta_probability"]}
        for item in ranked_nodes if item["delta_probability"] > 0
    ][:5]

    top_countervailing_nodes = [
        {"node_id": item["node_id"], "delta_probability": item["delta_probability"]}
        for item in reversed(ranked_nodes) if item["delta_probability"] < 0
    ][:5]

    return {
        "target_class": resolved_target,
        "target_label": modality_res["target_label"],
        "predicted_class": modality_res["predicted_class"],
        "predicted_label": modality_res["predicted_label"],
        "baseline_probability": modality_res["baseline_probability"],
        "modality_attributions": modality_res["modality_attributions"],
        "root_attribution": node_res["root_attribution"],
        "top_supporting_user_nodes": top_supporting_nodes,
        "top_countervailing_user_nodes": top_countervailing_nodes,
        "total_active_user_nodes": node_res["total_active_user_nodes"],
        "faithfulness": faithfulness_res,
        "methodological_notes": {
            "sign_convention": modality_res["sign_convention"],
            "score_interpretation": modality_res["score_interpretation"],
            "disclaimer": node_res["disclaimer"],
        },
    }
