"""
Propagation Driver Analysis Script.

Presentation-safe version.

The script:

1. Loads a trained STGNN checkpoint.
2. Predicts Fake/Real cascade veracity.
3. Computes cascade topology statistics.
4. Ranks propagation-driver candidates using structural
   rumor centrality.
5. Produces cascade visualizations.

IMPORTANT:
UPFD's node 0 is the news/root node, not a verified human
patient-zero label. Therefore this script does NOT report
patient-zero Hit@1/MRR or claim bot identification.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from src.data.dataset_loader import (
    load_upfd_data,
)
from src.data.temporal_builder import (
    TemporalCascadeBuilder,
)
from src.models.stgnn import (
    SpatioTemporalGNN,
)
from src.models.source_identifier import (
    CascadeSourceIdentifier,
)
from src.data.cascade_features import (
    build_networkx_tree,
    compute_cascade_metrics,
)
from src.visualization.cascade_visualizer import (
    plot_cascade_tree,
)

from scripts.train import (
    build_progression_snapshots,
    build_snapshots_for_run,
)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze propagation-driver candidates "
            "in misinformation cascades."
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
        "--checkpoint",
        type=str,
        default=(
            "./checkpoints/"
            "stgnn_politifact_best.pt"
        ),
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "./outputs/"
            "driver_analysis"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
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
        "--top_k",
        type=int,
        default=5,
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        args.device
    )

    print(
        "\n=========================================================="
    )

    print(
        "  PROPAGATION DRIVER ANALYSIS"
    )

    print(
        "=========================================================="
    )

    print(
        f"Dataset : UPFD-{args.dataset.capitalize()}"
    )

    print(
        f"Device  : {device}"
    )

    print(
        f"Samples : {args.num_samples}"
    )

    # -------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------

    _, _, test_ds, meta = (
        load_upfd_data(
            name=args.dataset,
            feature="bert",
            multimodal=True,
            use_synthetic_fallback=False,
            require_timestamps=(
                args.temporal_mode
                == "timestamp"
            ),
        )
    )

    in_dim = int(
        meta["input_dim"]
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
            "\nTemporal mode: topology-based progression."
        )

        print(
            "This is prototype progression, not real-time timestamp data."
        )

        temporal_builder = None

    else:

        print(
            "\nTemporal mode: official UPFD timestamps."
        )

        temporal_builder = (
            TemporalCascadeBuilder(
                num_snapshots=args.num_snapshots,
                add_temporal_features=True,
                require_complete_timestamps=True,
            )
        )

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    model = SpatioTemporalGNN(
        in_features=in_dim + 4,
        hidden_dim=64,
        num_snapshots=args.num_snapshots,
        temporal_cell="gru",
        num_classes=2,
    ).to(device)

    if not os.path.exists(
        args.checkpoint
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{args.checkpoint}\n\n"
            "Train the STGNN first using scripts/train.py."
        )

    print(
        f"\nLoading checkpoint:"
        f"\n{args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # -------------------------------------------------------------
    # Driver identifier
    # -------------------------------------------------------------

    source_identifier = (
        CascadeSourceIdentifier(
            hybrid_weight=0.7,
        )
    )

    sample_count = min(
        args.num_samples,
        len(test_ds),
    )

    print(
        f"\nAnalyzing {sample_count} test cascades..."
    )

    aggregate_results = []

    # -------------------------------------------------------------
    # Analyze individual cascades
    # -------------------------------------------------------------

    for idx in range(
        sample_count
    ):

        cascade = test_ds[idx]

        num_nodes = int(
            cascade.x.size(0)
        )

        cascade_label = (
            "FAKE"
            if int(cascade.y.item()) == 1
            else "REAL"
        )

        # ---------------------------------------------------------
        # Build progression/timestamp snapshots
        # ---------------------------------------------------------

        if resolved_temporal_mode == "timestamp":

            snapshots = (
                build_snapshots_for_run(
                    cascade,
                    temporal_builder,
                    "timestamp",
                )
            )

        else:

            snapshots = (
                build_progression_snapshots(
                    cascade,
                    args.num_snapshots,
                )
            )

        snapshots = [
            snapshot.to(device)
            for snapshot in snapshots
        ]

        # ---------------------------------------------------------
        # Model prediction
        # ---------------------------------------------------------

        with torch.no_grad():

            veracity_logits, node_scores, _ = (
                model.forward_snapshots(
                    snapshots
                )
            )

            probabilities = torch.softmax(
                veracity_logits,
                dim=-1,
            )

            predicted_class = int(
                torch.argmax(
                    probabilities,
                    dim=-1,
                ).item()
            )

        predicted_label = (
            "FAKE"
            if predicted_class == 1
            else "REAL"
        )

        confidence = float(
            probabilities[
                0,
                predicted_class,
            ].item()
        )

        # ---------------------------------------------------------
        # Structural propagation-driver ranking
        # ---------------------------------------------------------

        ranked_candidates = (
            source_identifier.rank_source_candidates(
                data=cascade,
                gnn_node_logits=None,
                method="rumor_centrality",
                top_k=args.top_k,
            )
        )

        top_driver = (
            ranked_candidates[0][0]
            if ranked_candidates
            else None
        )

        # ---------------------------------------------------------
        # Cascade metrics
        # ---------------------------------------------------------

        graph = build_networkx_tree(
            cascade.edge_index,
            num_nodes,
        )

        cascade_metrics = (
            compute_cascade_metrics(
                cascade.edge_index,
                num_nodes,
            )
        )

        # ---------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------

        image_path = os.path.join(
            args.output_dir,
            f"cascade_{idx}_{cascade_label.lower()}.png",
        )

        plot_cascade_tree(
            edge_index=cascade.edge_index,
            num_nodes=num_nodes,
            source_node=0,
            bot_nodes=[],
            predicted_source=(
                top_driver
                if top_driver is not None
                else 0
            ),
            title=(
                f"Cascade #{idx} | "
                f"Actual: {cascade_label} | "
                f"Predicted: {predicted_label} | "
                f"Confidence: {confidence:.2f}"
            ),
            save_path=image_path,
        )

        # ---------------------------------------------------------
        # Console report
        # ---------------------------------------------------------

        print(
            f"\n--- Cascade #{idx} ---"
        )

        print(
            f"Actual label        : {cascade_label}"
        )

        print(
            f"Model prediction    : {predicted_label}"
        )

        print(
            f"Prediction confidence: "
            f"{confidence:.4f}"
        )

        print(
            f"Nodes               : {num_nodes}"
        )

        print(
            f"Max depth           : "
            f"{cascade_metrics['max_depth']}"
        )

        print(
            f"Structural virality : "
            f"{cascade_metrics['structural_virality']:.3f}"
        )

        print(
            "\nPropagation Driver Candidates:"
        )

        for rank, (
            node_id,
            score,
        ) in enumerate(
            ranked_candidates,
            start=1,
        ):

            print(
                f"  #{rank} "
                f"Node {node_id:<4} "
                f"Driver Score = {score:.4f}"
            )

        print(
            "\nRoot node:"
        )

        print(
            "  Node 0 = News/Root Node"
        )

        print(
            "\nVisualization:"
        )

        print(
            f"  {image_path}"
        )

        # ---------------------------------------------------------
        # Store machine-readable result
        # ---------------------------------------------------------

        result = {
            "cascade_index": idx,
            "true_label": cascade_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "num_nodes": num_nodes,
            "temporal_mode": resolved_temporal_mode,
            "top_driver": (
                int(top_driver)
                if top_driver is not None
                else None
            ),
            "driver_candidates": [
                {
                    "node": int(node),
                    "score": float(score),
                }
                for node, score
                in ranked_candidates
            ],
            "cascade_metrics": {
                key: float(value)
                if isinstance(
                    value,
                    (int, float),
                )
                else value
                for key, value
                in cascade_metrics.items()
            },
            "visualization": image_path,
        }

        aggregate_results.append(
            result
        )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    output_json = os.path.join(
        args.output_dir,
        "driver_analysis.json",
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "dataset": args.dataset,
                "temporal_mode": resolved_temporal_mode,
                "num_samples": sample_count,
                "results": aggregate_results,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    print(
        "\n=========================================================="
    )

    print(
        "  DRIVER ANALYSIS COMPLETE"
    )

    print(
        "=========================================================="
    )

    print(
        f"Samples analyzed : {sample_count}"
    )

    print(
        "Source metric    : Not reported"
    )

    print(
        "Reason           : No verified UPFD patient-zero labels"
    )

    print(
        f"Results saved    : {output_json}"
    )


if __name__ == "__main__":
    main()