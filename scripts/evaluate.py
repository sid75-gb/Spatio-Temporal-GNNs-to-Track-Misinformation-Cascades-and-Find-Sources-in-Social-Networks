"""
Comparative benchmark:

    BERT
    Static GAT
    Spatio-Temporal GNN (STGNN)

Current presentation benchmark focuses on graph-level
Fake/Real classification.

Source-localization metrics are deliberately excluded because
UPFD does not provide verified human patient-zero labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root so "src" can be imported when this file
# is executed as: python scripts/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


try:
    from tabulate import tabulate

    HAS_TABULATE = True

except ImportError:

    HAS_TABULATE = False

from src.data.dataset_loader import (
    load_upfd_data,
    get_dataloaders,
)
from src.data.temporal_builder import (
    TemporalCascadeBuilder,
)
from scripts.train import (
    set_seed,
    build_model,
    train_epoch,
    evaluate_model,
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare BERT, Static GAT and STGNN "
            "on UPFD."
        )
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
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Benchmark a single model
# ---------------------------------------------------------------------

def benchmark_model(
    model_name,
    args,
    train_loader,
    val_loader,
    test_loader,
    in_dim,
    device,
    temporal_builder=None,
    temporal_mode="not_applicable",
):
    """
    Train one model and evaluate it on the test set.
    """

    print(
        "\n=================================================="
    )

    print(
        f"  TRAINING: {model_name.upper()}"
    )

    print(
        "=================================================="
    )

    # Each model gets the same random seed.
    set_seed(
        args.seed
    )

    model = build_model(
        model_type=model_name,
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_snapshots=args.num_snapshots,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion_clf = nn.CrossEntropyLoss()

    best_val_f1 = -float(
        "inf"
    )

    best_weights = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        train_metrics = train_epoch(
            model=model,
            model_type=model_name,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion_clf=criterion_clf,
            temporal_builder=temporal_builder,
            device=device,
            temporal_mode=temporal_mode,
        )

        val_metrics = evaluate_model(
            model=model,
            model_type=model_name,
            dataloader=val_loader,
            criterion_clf=criterion_clf,
            temporal_builder=temporal_builder,
            device=device,
            temporal_mode=temporal_mode,
        )

        current_f1 = float(
            val_metrics["f1_macro"]
        )

        if current_f1 > best_val_f1:

            best_val_f1 = current_f1

            best_weights = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == args.epochs
        ):

            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"Train Acc {train_metrics['accuracy']:.3f} | "
                f"Val Acc {val_metrics['accuracy']:.3f} | "
                f"Val F1 {val_metrics['f1_macro']:.3f}",
                flush=True,
            )

    if best_weights is None:
        raise RuntimeError(
            f"No valid checkpoint found for {model_name}."
        )

    model.load_state_dict(
        {
            key: value.to(device)
            for key, value
            in best_weights.items()
        }
    )

    test_metrics = evaluate_model(
        model=model,
        model_type=model_name,
        dataloader=test_loader,
        criterion_clf=criterion_clf,
        temporal_builder=temporal_builder,
        device=device,
        temporal_mode=temporal_mode,
    )

    return test_metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print(
        "\n================================================================"
    )

    print(
        "  UPFD MISINFORMATION MODEL COMPARISON"
    )

    print(
        "================================================================"
    )

    print(
        f"Dataset       : UPFD-{args.dataset.capitalize()}"
    )

    print(
        f"Epochs        : {args.epochs}"
    )

    print(
        f"Batch size    : {args.batch_size}"
    )

    print(
        f"Temporal mode : {args.temporal_mode}"
    )

    print(
        f"Device        : {device}"
    )

    # -------------------------------------------------------------
    # 1. NLP dataset
    # -------------------------------------------------------------

    print(
        "\nLoading BERT-only dataset..."
    )

    (
        nlp_train_ds,
        nlp_val_ds,
        nlp_test_ds,
        nlp_meta,
    ) = load_upfd_data(
        name=args.dataset,
        feature="bert",
        multimodal=False,
        batch_size=args.batch_size,
        use_synthetic_fallback=False,
        require_timestamps=False,
    )

    (
        nlp_train_loader,
        nlp_val_loader,
        nlp_test_loader,
    ) = get_dataloaders(
        nlp_train_ds,
        nlp_val_ds,
        nlp_test_ds,
        batch_size=args.batch_size,
    )

    # -------------------------------------------------------------
    # 2. Multimodal graph dataset
    # -------------------------------------------------------------

    print(
        "Loading multimodal graph dataset..."
    )

    (
        graph_train_ds,
        graph_val_ds,
        graph_test_ds,
        graph_meta,
    ) = load_upfd_data(
        name=args.dataset,
        feature="bert",
        multimodal=True,
        batch_size=args.batch_size,
        use_synthetic_fallback=False,
        require_timestamps=(
            args.temporal_mode
            == "timestamp"
        ),
    )

    (
        graph_train_loader,
        graph_val_loader,
        graph_test_loader,
    ) = get_dataloaders(
        graph_train_ds,
        graph_val_ds,
        graph_test_ds,
        batch_size=args.batch_size,
    )

    in_dim = int(
        graph_meta["input_dim"]
    )

    timestamp_available = bool(
        graph_meta.get(
            "timestamp_mapping_available",
            False,
        )
    )

    # -------------------------------------------------------------
    # Resolve temporal mode
    # -------------------------------------------------------------

    if args.temporal_mode == "timestamp":

        resolved_temporal_mode = (
            "timestamp"
        )

    elif (
        args.temporal_mode == "auto"
        and timestamp_available
    ):

        resolved_temporal_mode = (
            "timestamp"
        )

    else:

        resolved_temporal_mode = (
            "progression"
        )

    if resolved_temporal_mode == "progression":

        print(
            "\nNOTE:"
        )

        print(
            "Official UPFD timestamps are unavailable."
        )

        print(
            "STGNN benchmark will use topology-based progression phases."
        )

    if resolved_temporal_mode == "timestamp":
        temporal_builder = TemporalCascadeBuilder(
            num_snapshots=args.num_snapshots,
            add_temporal_features=True,
            require_complete_timestamps=True,
        )
    else:
        # Builder object is still required by the shared training/evaluation
        # functions. In progression mode, the topology-based snapshot
        # generator is used instead of real timestamps.
        temporal_builder = TemporalCascadeBuilder(
            num_snapshots=args.num_snapshots,
            add_temporal_features=True,
            require_complete_timestamps=False,
        )

    # -------------------------------------------------------------
    # Run models
    # -------------------------------------------------------------

    results = {}

    # BERT
    results["nlp"] = benchmark_model(
        model_name="nlp",
        args=args,
        train_loader=nlp_train_loader,
        val_loader=nlp_val_loader,
        test_loader=nlp_test_loader,
        in_dim=768,
        device=device,
        temporal_builder=None,
        temporal_mode="not_applicable",
    )

    # Static GAT
    results["static_gnn"] = benchmark_model(
        model_name="static_gnn",
        args=args,
        train_loader=graph_train_loader,
        val_loader=graph_val_loader,
        test_loader=graph_test_loader,
        in_dim=in_dim,
        device=device,
        temporal_builder=None,
        temporal_mode="not_applicable",
    )

    # STGNN
    results["stgnn"] = benchmark_model(
        model_name="stgnn",
        args=args,
        train_loader=graph_train_loader,
        val_loader=graph_val_loader,
        test_loader=graph_test_loader,
        in_dim=in_dim,
        device=device,
        temporal_builder=temporal_builder,
        temporal_mode=resolved_temporal_mode,
    )

    # -------------------------------------------------------------
    # Table
    # -------------------------------------------------------------

    headers = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]

    labels = {
        "nlp": "BERT NLP",
        "static_gnn": "Static GAT",
        "stgnn": "STGNN",
    }

    rows = []

    for model_name in [
        "nlp",
        "static_gnn",
        "stgnn",
    ]:

        metrics = results[
            model_name
        ]

        rows.append(
            [
                labels[model_name],
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision_macro']:.4f}",
                f"{metrics['recall_macro']:.4f}",
                f"{metrics['f1_macro']:.4f}",
                f"{metrics['roc_auc']:.4f}",
            ]
        )

    print(
        "\n================================================================"
    )

    print(
        "  COMPARATIVE BENCHMARK RESULTS"
    )

    print(
        "================================================================"
    )

    if HAS_TABULATE:

        print(
            tabulate(
                rows,
                headers=headers,
                tablefmt="fancy_grid",
            )
        )

    else:

        print(
            " | ".join(headers)
        )

        print(
            "-" * 90
        )

        for row in rows:

            print(
                " | ".join(
                    str(value)
                    for value in row
                )
            )

    # -------------------------------------------------------------
    # F1 improvement
    # -------------------------------------------------------------

    bert_f1 = float(
        results["nlp"]["f1_macro"]
    )

    stgnn_f1 = float(
        results["stgnn"]["f1_macro"]
    )

    improvement = (
        (
            stgnn_f1
            - bert_f1
        )
        / max(
            1e-8,
            abs(bert_f1),
        )
        * 100.0
    )

    print(
        f"\nSTGNN F1 change vs BERT: "
        f"{improvement:+.2f}%"
    )

    # -------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------

    result_path = os.path.join(
        args.output_dir,
        f"benchmark_{args.dataset}.json",
    )

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "temporal_mode": resolved_temporal_mode,
        "results": results,
    }

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    print(
        f"\nBenchmark results saved to:"
        f"\n{result_path}"
    )


if __name__ == "__main__":
    main()