"""
Training pipeline for:

1. NLP content baseline
2. Static GAT cascade baseline
3. Spatio-Temporal GNN (STGNN)

Presentation-safe version:

- Uses real UPFD graph data.
- Does not silently fall back to synthetic data.
- Does not use node 0 as a fake source ground truth.
- Supports explicit topology-based progression when official
  timestamp mappings are unavailable.

IMPORTANT:
"progression" mode is a presentation/prototype mode, not real-time
timestamp modeling. For final research experiments, use "timestamp"
mode after obtaining the official UPFD timestamp mappings.
"""

from __future__ import annotations
from torch.optim import optimizer

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

from torch_geometric.data import Data

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from src.data.dataset_loader import (
    load_upfd_data,
    get_dataloaders,
)
from src.data.temporal_builder import (
    TemporalCascadeBuilder,
)
from src.models.nlp_baseline import (
    NLPContentClassifier,
)
from src.models.static_gnn import (
    StaticCascadeGNN,
)
from src.models.stgnn import (
    SpatioTemporalGNN,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Presentation-mode temporal snapshots
# ---------------------------------------------------------------------

def build_progression_snapshots(
    data: Data,
    num_snapshots: int = 4,
) -> List[Data]:
    """
    Build topology-based cascade progression snapshots.

    IMPORTANT:
    This is NOT timestamp-based temporal modeling.

    It creates:
        P1 -> P2 -> P3 -> P4

    by progressively revealing nodes according to propagation-tree
    structure.

    This is used only because the local project currently lacks the
    official UPFD timestamp mapping files.
    """

    num_nodes = int(data.x.size(0))

    if num_nodes < 2:
        raise ValueError(
            "Cascade must contain at least the news root and one user."
        )

    edge_index = data.edge_index

    # Build undirected view only for computing propagation depth.
    graph = nx.DiGraph()

    graph.add_nodes_from(
        range(num_nodes)
    )

    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()

        graph.add_edges_from(
            zip(src, dst)
        )

    undirected_graph = graph.to_undirected()

    # Root/news node is 0.
    depths = nx.single_source_shortest_path_length(
        undirected_graph,
        0,
    )

    # Reveal users according to graph depth and node index.
    user_nodes = list(
        range(1, num_nodes)
    )

    user_nodes.sort(
        key=lambda node: (
            depths.get(node, 999999),
            node,
        )
    )

    if num_snapshots == 4:
        fractions = [0.25, 0.50, 0.75, 1.00]

    elif num_snapshots == 5:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    else:
        fractions = [
            float(x)
            for x in np.linspace(
                1.0 / num_snapshots,
                1.0,
                num_snapshots,
            )
        ]
    snapshots: List[Data] = []

    for step_idx, fraction in enumerate(
        fractions
    ):
        num_active_users = max(
            1,
            int(
                np.ceil(
                    fraction
                    * len(user_nodes)
                )
            ),
        )

        num_active_users = min(
            num_active_users,
            len(user_nodes),
        )

        active_users = user_nodes[
            :num_active_users
        ]

        active_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool,
        )

        # News/root is always present.
        active_mask[0] = True

        if active_users:
            active_mask[
                torch.tensor(
                    active_users,
                    dtype=torch.long,
                )
            ] = True

        # Keep only edges among observed nodes.
        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]

            edge_mask = (
                active_mask[src]
                & active_mask[dst]
            )

            snapshot_edges = (
                edge_index[:, edge_mask]
            )
        else:
            snapshot_edges = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        # Current observed graph.
        current_graph = nx.DiGraph()

        current_nodes = torch.where(
            active_mask
        )[0].tolist()

        current_graph.add_nodes_from(
            current_nodes
        )

        if snapshot_edges.numel() > 0:
            current_src = (
                snapshot_edges[0]
                .detach()
                .cpu()
                .tolist()
            )

            current_dst = (
                snapshot_edges[1]
                .detach()
                .cpu()
                .tolist()
            )

            current_graph.add_edges_from(
                zip(
                    current_src,
                    current_dst,
                )
            )

        current_undirected = (
            current_graph.to_undirected()
        )

        current_depths = (
            nx.single_source_shortest_path_length(
                current_undirected,
                0,
            )
            if 0 in current_undirected
            else {0: 0}
        )

        max_depth = max(
            current_depths.values(),
            default=0,
        )

        current_out_degree = dict(
            current_graph.out_degree()
        )

        max_out_degree = max(
            current_out_degree.values(),
            default=0,
        )

        # Four temporal/progression features:
        #
        # 1. current normalized depth
        # 2. current normalized out-degree
        # 3. progression fraction
        # 4. active flag
        progression_features = torch.zeros(
            num_nodes,
            4,
            dtype=torch.float32,
        )

        for node in current_nodes:

            depth_value = float(
                current_depths.get(
                    node,
                    0,
                )
            )

            degree_value = float(
                current_out_degree.get(
                    node,
                    0,
                )
            )

            normalized_depth = (
                depth_value / max_depth
                if max_depth > 0
                else 0.0
            )

            normalized_degree = (
                degree_value / max_out_degree
                if max_out_degree > 0
                else 0.0
            )

            progression_features[node] = torch.tensor(
                [
                    normalized_depth,
                    normalized_degree,
                    float(fraction),
                    1.0,
                ],
                dtype=torch.float32,
            )

        node_x = data.x.float().clone()

        node_x = torch.cat(
            [
                node_x,
                progression_features,
            ],
            dim=-1,
        )

        # Future/inactive nodes contain no feature information.
        node_x[~active_mask] = 0.0

        snapshot = Data(
            x=node_x,
            edge_index=snapshot_edges,
            y=data.y.clone()
            if data.y is not None
            else None,
            active_mask=active_mask,
            snapshot_step=torch.tensor(
                step_idx,
                dtype=torch.long,
            ),
            snapshot_fraction=torch.tensor(
                float(fraction),
                dtype=torch.float32,
            ),
            progression_mode=torch.tensor(
                True,
                dtype=torch.bool,
            ),
            num_nodes=num_nodes,
        )

        if hasattr(
            data,
            "global_node_ids",
        ):
            snapshot.global_node_ids = (
                data.global_node_ids.clone()
            )

        snapshots.append(
            snapshot
        )

    return snapshots


# ---------------------------------------------------------------------
# Temporal snapshot dispatcher
# ---------------------------------------------------------------------

def build_snapshots_for_run(
    data: Data,
    temporal_builder: TemporalCascadeBuilder,
    temporal_mode: str,
) -> List[Data]:
    """
    Build snapshots according to selected mode.

    Modes:

        progression:
            topology-based presentation progression

        timestamp:
            real UPFD timestamp-based snapshots

        auto:
            automatically select timestamp mode if real timestamps
            exist; otherwise use progression mode.
    """

    temporal_mode = temporal_mode.lower()

    if temporal_mode == "progression":
        return build_progression_snapshots(
            data,
            temporal_builder.num_snapshots,
        )

    if temporal_mode == "timestamp":
        return temporal_builder.build_snapshots(
            data
        )

    if temporal_mode == "auto":
        timestamps = getattr(
            data,
            "node_timestamps",
            None,
        )

        if timestamps is not None:
            user_timestamps = timestamps[1:]

            if user_timestamps.numel() > 0:
                valid = torch.isfinite(
                    user_timestamps
                )

                if bool(
                    valid.all()
                ):
                    return temporal_builder.build_snapshots(
                        data
                    )

        return build_progression_snapshots(
            data,
            temporal_builder.num_snapshots,
        )

    raise ValueError(
        f"Unknown temporal_mode='{temporal_mode}'. "
        "Use auto, progression, or timestamp."
    )


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train misinformation detection models "
            "on the UPFD benchmark."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="stgnn",
        choices=[
            "stgnn",
            "static_gnn",
            "nlp",
        ],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="politifact",
        choices=[
            "politifact",
            "gossipcop",
        ],
    )

    parser.add_argument(
        "--feature",
        type=str,
        default="bert",
    )

    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use BERT + profile features for graph models.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
    "--weight_decay",
    type=float,
    default=1e-4,
)

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=8,
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_snapshots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--temporal_mode",
        type=str,
        default="auto",
        choices=[
            "auto",
            "progression",
            "timestamp",
        ],
        help=(
            "auto = timestamps if available, otherwise "
            "topology progression."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save training result JSON (default: ./outputs).",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Custom output filename for training result JSON.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing checkpoint or output result files.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------

def train_epoch(
    model,
    model_type,
    dataloader,
    optimizer,
    criterion_clf,
    temporal_builder=None,
    device=torch.device("cpu"),
    temporal_mode="progression",
):
    """
    Train one epoch.

    Source localization is deliberately NOT used as a supervised loss
    because UPFD does not provide a verified human patient-zero label.
    """

    model.train()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in dataloader:

        batch = batch.to(
            device
        )

        optimizer.zero_grad()

        # -------------------------------------------------------------
        # NLP
        # -------------------------------------------------------------
        if model_type == "nlp":

            logits = model(
                batch
            )

            loss = criterion_clf(
                logits,
                batch.y,
            )

        # -------------------------------------------------------------
        # Static GNN
        # -------------------------------------------------------------
        elif model_type == "static_gnn":

            logits = model(
                batch
            )

            loss = criterion_clf(
                logits,
                batch.y,
            )

        # -------------------------------------------------------------
        # STGNN
        # -------------------------------------------------------------
        elif model_type == "stgnn":

            if temporal_builder is None:
                raise ValueError(
                    "temporal_builder is required for STGNN."
                )

            cascade_list = (
                batch.to_data_list()
            )

            batch_logits = []

            for cascade in cascade_list:

                snapshots = (
                    build_snapshots_for_run(
                        cascade,
                        temporal_builder,
                        temporal_mode,
                    )
                )

                snapshots = [
                    snapshot.to(device)
                    for snapshot in snapshots
                ]

                cascade_logits, _, _ = (
                    model.forward_snapshots(
                        snapshots
                    )
                )

                batch_logits.append(
                    cascade_logits
                )

            logits = torch.cat(
                batch_logits,
                dim=0,
            )

            loss = criterion_clf(
                logits,
                batch.y,
            )

        else:
            raise ValueError(
                f"Unknown model_type={model_type}"
            )

        # -------------------------------------------------------------
        # Optimization
        # -------------------------------------------------------------
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        preds = torch.argmax(
            logits,
            dim=-1,
        )

        all_preds.extend(
            preds.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            batch.y.detach()
            .cpu()
            .tolist()
        )

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
    )

    metrics["loss"] = (
        total_loss
        / len(dataloader.dataset)
    )

    return metrics


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    model_type,
    dataloader,
    criterion_clf,
    temporal_builder=None,
    device=torch.device("cpu"),
    temporal_mode="progression",
):
    """
    Evaluate classification performance only.

    Source-localization metrics are intentionally excluded because
    the current UPFD setup does not have verified patient-zero labels.
    """

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:

        batch = batch.to(
            device
        )

        # -------------------------------------------------------------
        # NLP / Static
        # -------------------------------------------------------------
        if model_type in [
            "nlp",
            "static_gnn",
        ]:

            logits = model(
                batch
            )

            loss = criterion_clf(
                logits,
                batch.y,
            )

            probs = F.softmax(
                logits,
                dim=-1,
            )

        # -------------------------------------------------------------
        # STGNN
        # -------------------------------------------------------------
        elif model_type == "stgnn":

            if temporal_builder is None:
                raise ValueError(
                    "temporal_builder is required for STGNN."
                )

            cascade_list = (
                batch.to_data_list()
            )

            batch_logits = []

            for cascade in cascade_list:

                snapshots = (
                    build_snapshots_for_run(
                        cascade,
                        temporal_builder,
                        temporal_mode,
                    )
                )

                snapshots = [
                    snapshot.to(device)
                    for snapshot in snapshots
                ]

                cascade_logits, _, _ = (
                    model.forward_snapshots(
                        snapshots
                    )
                )

                batch_logits.append(
                    cascade_logits
                )

            logits = torch.cat(
                batch_logits,
                dim=0,
            )

            loss = criterion_clf(
                logits,
                batch.y,
            )

            probs = F.softmax(
                logits,
                dim=-1,
            )

        else:
            raise ValueError(
                f"Unknown model_type={model_type}"
            )

        total_loss += (
            loss.item()
            * batch.num_graphs
        )

        preds = torch.argmax(
            logits,
            dim=-1,
        )

        all_preds.extend(
            preds.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            batch.y.detach()
            .cpu()
            .tolist()
        )

        all_probs.extend(
            probs.detach()
            .cpu()
            .tolist()
        )

    metrics = compute_classification_metrics(
        all_labels,
        all_preds,
        y_probs=torch.tensor(
            all_probs,
            dtype=torch.float32,
        ),
    )

    metrics["loss"] = (
        total_loss
        / len(dataloader.dataset)
    )

    return metrics


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_model(
    model_type: str,
    in_dim: int,
    hidden_dim: int,
    num_snapshots: int,
    device: torch.device,
):
    """Construct the requested model."""

    if model_type == "nlp":

        model = NLPContentClassifier(
            in_features=768,
            hidden_dim=hidden_dim,
            num_classes=2,
        )

    elif model_type == "static_gnn":

        model = StaticCascadeGNN(
            in_features=in_dim,
            hidden_dim=hidden_dim,
            gnn_type="gat",
            num_classes=2,
        )

    elif model_type == "stgnn":

        model = SpatioTemporalGNN(
            in_features=in_dim + 4,
            hidden_dim=hidden_dim,
            num_snapshots=num_snapshots,
            temporal_cell="gru",
            num_classes=2,
        )

    else:
        raise ValueError(
            f"Unknown model_type={model_type}"
        )

    return model.to(device)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    # Determine intended best-checkpoint path and result JSON path before training
    checkpoint_path = os.path.join(
        args.save_dir,
        f"{args.model}_{args.dataset}_best.pt",
    )

    if args.output_file:
        result_path = os.path.join(
            args.output_dir,
            args.output_file,
        )
    else:
        result_path = os.path.join(
            args.output_dir,
            f"{args.model}_{args.dataset}_results.json",
        )

    # Abort before training starts if either destination already exists and --overwrite is not supplied
    if not args.overwrite:
        conflicts = []
        if os.path.exists(checkpoint_path):
            conflicts.append(f"Checkpoint: {checkpoint_path}")
        if os.path.exists(result_path):
            conflicts.append(f"Result JSON: {result_path}")

        if conflicts:
            print("\n" + "=" * 65)
            print("  [ABORTED] Destination artifact(s) already exist:")
            for conflict in conflicts:
                print(f"    - {conflict}")
            print("  Safety constraint: Existing artifacts cannot be overwritten by default.")
            print("  Use --overwrite to explicitly permit replacing existing files.")
            print("=" * 65 + "\n")
            sys.exit(1)

    set_seed(
        args.seed
    )

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    print(
        "\n=========================================================="
    )
    print(
        "  MISINFORMATION DETECTION TRAINING PIPELINE"
    )
    print(
        "=========================================================="
    )

    print(
        f"Model       : {args.model.upper()}"
    )
    print(
        f"Dataset     : UPFD-{args.dataset.capitalize()}"
    )
    print(
        f"Device      : {device}"
    )
    print(
        f"Seed        : {args.seed}"
    )
    print(
        f"Temporal    : {args.temporal_mode}"
    )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    use_multimodal = (
        args.multimodal
        and args.model != "nlp"
    )

    require_timestamps = (
        args.temporal_mode
        == "timestamp"
    )

    train_ds, val_ds, test_ds, meta = (
        load_upfd_data(
            name=args.dataset,
            feature=args.feature,
            multimodal=use_multimodal,
            batch_size=args.batch_size,
            use_synthetic_fallback=False,
            require_timestamps=require_timestamps,
        )
    )

    train_loader, val_loader, test_loader = (
        get_dataloaders(
            train_ds,
            val_ds,
            test_ds,
            batch_size=args.batch_size,
        )
    )

    in_dim = int(
        meta["input_dim"]
    )

    print(
        f"\nTrain graphs : {meta['train_size']}"
    )
    print(
        f"Val graphs   : {meta['val_size']}"
    )
    print(
        f"Test graphs  : {meta['test_size']}"
    )
    print(
        f"Input dim    : {in_dim}"
    )

    timestamp_available = bool(
        meta.get(
            "timestamp_mapping_available",
            False,
        )
    )

    # -------------------------------------------------------------
    # Resolve temporal mode
    # -------------------------------------------------------------

    if args.model == "stgnn":

        if args.temporal_mode == "timestamp":
            resolved_temporal_mode = "timestamp"

        elif (
            args.temporal_mode == "auto"
            and timestamp_available
        ):
            resolved_temporal_mode = "timestamp"

        else:
            resolved_temporal_mode = "progression"

        if resolved_temporal_mode == "progression":
            print("\nNOTE:")
            print(
                "Using topology-based cascade progression for this baseline."
            )

            if timestamp_available:
                print(
                    "Official timestamps are available but intentionally "
                    "not used in this run."
                )
            else:
                print(
                    "Official timestamp mappings are unavailable."
                )

        # Always create the builder object.
        # In progression mode, train_epoch/evaluate_model use the
        # topology-based snapshot function instead of real timestamps.
        temporal_builder = TemporalCascadeBuilder(
            num_snapshots=args.num_snapshots,
            add_temporal_features=True,
            require_complete_timestamps=(
                resolved_temporal_mode == "timestamp"
            ),
        )

    else:

        resolved_temporal_mode = "not_applicable"
        temporal_builder = None


    # -------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------

    model = build_model(
        model_type=args.model,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_snapshots=args.num_snapshots,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        threshold=1e-3,
    min_lr=1e-5,
)

    criterion_clf = nn.CrossEntropyLoss()

    best_val_f1 = -float(
        "inf"
    )
    epochs_without_improvement = 0

    checkpoint_path = os.path.join(
        args.save_dir,
        f"{args.model}_{args.dataset}_best.pt",
    )

    print(
        "\n--- Training Progress ---"
    )

    start_time = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_epoch(
            model=model,
            model_type=args.model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_clf=criterion_clf,
            temporal_builder=temporal_builder,
            device=device,
            temporal_mode=resolved_temporal_mode,
        )

        val_metrics = evaluate_model(
            model=model,
            model_type=args.model,
            dataloader=val_loader,
            criterion_clf=criterion_clf,
            temporal_builder=temporal_builder,
            device=device,
            temporal_mode=resolved_temporal_mode,
        )

        val_f1 = float(
            val_metrics["f1_macro"]
        )

        # Reduce the learning rate when validation F1 plateaus.
        scheduler.step(val_f1)

        is_best = (
            val_f1 > best_val_f1
        )

        if is_best:

            best_val_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "model_type": args.model,
                    "dataset": args.dataset,
                    "in_dim": in_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_snapshots": args.num_snapshots,
                    "temporal_mode": resolved_temporal_mode,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        marker = "* BEST" if is_best else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train F1 {train_metrics['f1_macro']:.3f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.3f} | "
            f"Val F1 {val_f1:.3f} {marker}",
            flush=True,
        )

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "\nEarly stopping: validation F1 did not improve "
                f"for {args.early_stopping_patience} consecutive epochs."
            )
            break

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining completed in {elapsed:.1f} seconds."
    )

    # -------------------------------------------------------------
    # Load best checkpoint
    # -------------------------------------------------------------

    if not os.path.exists(
        checkpoint_path
    ):
        raise RuntimeError(
            "No checkpoint was saved. "
            "Training did not produce a valid best model."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # -------------------------------------------------------------
    # Final test evaluation
    # -------------------------------------------------------------

    test_metrics = evaluate_model(
        model=model,
        model_type=args.model,
        dataloader=test_loader,
        criterion_clf=criterion_clf,
        temporal_builder=temporal_builder,
        device=device,
        temporal_mode=resolved_temporal_mode,
    )

    print(
        "\n=========================================================="
    )
    print(
        "  FINAL TEST RESULTS"
    )
    print(
        "=========================================================="
    )

    print(
        f"Accuracy    : {test_metrics['accuracy']:.4f}"
    )
    print(
        f"Precision   : {test_metrics['precision_macro']:.4f}"
    )
    print(
        f"Recall      : {test_metrics['recall_macro']:.4f}"
    )
    print(
        f"F1-Score    : {test_metrics['f1_macro']:.4f}"
    )
    print(
        f"ROC-AUC     : {test_metrics['roc_auc']:.4f}"
    )

    # -------------------------------------------------------------
    # Save JSON result
    # -------------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    if args.output_file:
        result_path = os.path.join(
            args.output_dir,
            args.output_file,
        )
    else:
        result_path = os.path.join(
            args.output_dir,
            f"{args.model}_{args.dataset}_results.json",
        )

    result_payload = {
        "model": args.model,
        "dataset": args.dataset,
        "temporal_mode": resolved_temporal_mode,
        "seed": args.seed,
        "metrics": test_metrics,
    }

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result_payload,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to: {result_path}"
    )

    return test_metrics


if __name__ == "__main__":
    main()