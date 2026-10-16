"""
Interactive Cascade Dashboard Server.

MSc Data Science Research Dashboard:
Supports BOTH UPFD benchmark datasets:
    - UPFD PolitiFact
    - UPFD GossipCop

Features:
    - Real saved STGNN checkpoints trained in timestamp mode
    - Standardized 10-epoch benchmark results (BERT NLP, Static GAT, STGNN)
    - Extended STGNN training results (early stopping, best val)
    - True UPFD timestamp-based chronological quantile snapshots
    - Exact Unix timestamps and UTC human-readable datetime progression
    - Node-level observed timestamps with Node Inspector integration
    - Structural propagation-driver identification (rumor centrality)
    - Heuristic amplification anomaly detection (coordination fanout)
    - Interactive Vis.js propagation tree and force layout
    - Explicit actual vs predicted label comparison and correctness status
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.server
import json
import os
import socket
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_upfd_data
from src.models.stgnn import SpatioTemporalGNN
from src.models.source_identifier import CascadeSourceIdentifier
from src.data.cascade_features import (
    build_networkx_tree,
    compute_cascade_metrics,
)
from src.algorithms.bot_detection import BotNetworkIsolator
from src.data.temporal_builder import TemporalCascadeBuilder


# ---------------------------------------------------------------------
# Default Paths
# ---------------------------------------------------------------------

DEFAULT_CHECKPOINTS = {
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

BENCHMARK_PATHS = {
    "politifact": PROJECT_ROOT / "outputs" / "benchmark_politifact.json",
    "gossipcop": PROJECT_ROOT / "outputs" / "benchmark_gossipcop.json",
}

EXTENDED_PATHS = {
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

DRIVER_ANALYSIS_PATHS = {
    "politifact": (
        PROJECT_ROOT
        / "outputs"
        / "source_analysis"
        / "politifact_timestamp"
        / "driver_analysis.json"
    ),
    "gossipcop": (
        PROJECT_ROOT
        / "outputs"
        / "source_analysis"
        / "gossipcop_timestamp"
        / "driver_analysis.json"
    ),
}

EARLY_STAGE_PATHS = {
    "politifact": (
        PROJECT_ROOT
        / "outputs"
        / "early_stage_politifact_stgnn.json"
    ),
    "gossipcop": (
        PROJECT_ROOT
        / "outputs"
        / "early_stage_gossipcop_stgnn.json"
    ),
}



# ---------------------------------------------------------------------
# Port handling
# ---------------------------------------------------------------------

def find_available_port(
    preferred_port: int,
    max_attempts: int = 20,
) -> int:
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    return preferred_port


# ---------------------------------------------------------------------
# Timestamp formatting helper
# ---------------------------------------------------------------------

def format_timestamp_utc(ts: Optional[float]) -> Optional[str]:
    """
    Format a Unix epoch timestamp as a human-readable UTC string:
        YYYY-MM-DD HH:MM:SS UTC
    Returns None if timestamp is None, non-finite, or <= 0.
    """
    if ts is None:
        return None
    try:
        val = float(ts)
        if not np.isfinite(val) or val <= 0:
            return None
        return datetime.fromtimestamp(val, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OSError, OverflowError):
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a clean human-readable string."""
    if seconds <= 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------
# Strict JSON-safe converter
# ---------------------------------------------------------------------

def to_json_value(value: Any):
    """
    Recursively convert values into standard JSON-compliant values.
    Converts NaN, +Infinity and -Infinity to None (null in JSON).
    """
    if value is None:
        return None

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, np.floating):
        val = float(value)
        return val if np.isfinite(val) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if torch.is_tensor(value):
        if value.numel() == 1:
            scalar = value.detach().cpu().item()
            if isinstance(scalar, (float, np.floating)):
                scalar = float(scalar)
                return scalar if np.isfinite(scalar) else None
            return scalar
        return to_json_value(value.detach().cpu().tolist())

    if isinstance(value, dict):
        return {str(k): to_json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_value(v) for v in value]

    return str(value)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_dashboard_model(
    checkpoint_path: Path,
    in_dim: int,
    num_snapshots: int = 4,
):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\nSTGNN checkpoint not found: {checkpoint_path}\n"
            "Please ensure the trained checkpoint is present."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # In SpatioTemporalGNN, in_features is in_dim + 4 temporal features
    model = SpatioTemporalGNN(
        in_features=in_dim + 4,
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_snapshots=checkpoint.get("num_snapshots", num_snapshots),
        temporal_cell="gru",
        num_classes=2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------
# Phase-specific prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_at_phase(
    model,
    snapshots: List,
    phase_index: int,
):
    """
    Predict using only the currently observed chronological quantile snapshot.
    The current snapshot is repeated for remaining positions to match
    the expected sequence length without peeking into future data.
    """
    phase_index = max(0, min(phase_index, len(snapshots) - 1))
    current_snapshot = snapshots[phase_index]

    partial_sequence = list(snapshots[: phase_index + 1])
    while len(partial_sequence) < len(snapshots):
        partial_sequence.append(current_snapshot.clone())

    logits, _, _ = model.forward_snapshots(partial_sequence)
    probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())
    fake_prob = float(probabilities[1].item())
    real_prob = float(probabilities[0].item())

    return {
        "predicted_label": "FAKE" if predicted_class == 1 else "REAL",
        "predicted_class": predicted_class,
        "confidence": confidence if np.isfinite(confidence) else None,
        "fake_probability": fake_prob if np.isfinite(fake_prob) else None,
        "real_probability": real_prob if np.isfinite(real_prob) else None,
    }


# ---------------------------------------------------------------------
# Build node & edge records
# ---------------------------------------------------------------------

def build_node_records(
    cascade,
    graph: nx.DiGraph,
    driver_scores: Dict[int, float],
    anomaly_scores: Dict[int, float],
    suspicious_nodes: List[int],
    active_mask: torch.Tensor,
    driver_nodes: Optional[List[int]] = None,
):
    active_nodes = set(torch.where(active_mask)[0].tolist())
    suspicious_set = set(suspicious_nodes)
    driver_set = set(driver_nodes or [])

    try:
        depths = nx.single_source_shortest_path_length(
            graph.to_undirected(),
            0,
        )
    except Exception:
        depths = {0: 0}

    node_timestamps = getattr(cascade, "node_timestamps", None)

    records = []
    num_nodes = cascade.x.size(0)

    for node in range(num_nodes):
        if node not in active_nodes:
            continue

        is_root = (node == 0)
        is_anomalous = (node in suspicious_set)
        is_driver = (node in driver_set)

        if is_root:
            role = "News / Root Node"
        elif is_anomalous and is_driver:
            role = "Driver & Amplifier (Dual Role)"
        elif is_driver:
            role = "Propagation Driver Candidate"
        elif is_anomalous:
            role = "Propagation Amplifier (Heuristic)"
        else:
            role = "Retweeter"

        out_degree = int(graph.out_degree(node)) if node in graph else 0

        d_score = float(driver_scores.get(node, 0.0))
        a_score = float(anomaly_scores.get(node, 0.0))

        # Node-level timestamp resolution
        node_ts: Optional[float] = None
        node_dt: Optional[str] = None
        if not is_root and node_timestamps is not None and node < node_timestamps.numel():
            raw_t = node_timestamps[node].item()
            if np.isfinite(raw_t) and raw_t > 0:
                node_ts = float(raw_t)
                node_dt = format_timestamp_utc(node_ts)

        if is_root:
            ts_label = "Not applicable (News root)"
        elif node_dt is not None:
            ts_label = node_dt
        else:
            ts_label = "Timestamp unavailable"

        records.append({
            "id": int(node),
            "label": "NEWS" if is_root else f"U{node}",
            "role": role,
            "depth": int(depths.get(node, 0)),
            "out_degree": out_degree,
            "driver_score": d_score if np.isfinite(d_score) else 0.0,
            "anomaly_score": a_score if np.isfinite(a_score) else 0.0,
            "is_root": is_root,
            "is_driver": is_driver,
            "is_amplifier": is_anomalous,
            "timestamp": node_ts,
            "datetime_utc": node_dt,
            "timestamp_label": ts_label,
        })

    return records


def build_edge_records(
    edge_index: torch.Tensor,
    active_mask: torch.Tensor,
):
    active = set(torch.where(active_mask)[0].tolist())
    edges = []

    if edge_index is None or edge_index.numel() == 0:
        return edges

    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()

    for parent, child in zip(src, dst):
        if parent in active and child in active:
            edges.append({
                "from": int(parent),
                "to": int(child),
            })

    return edges


# ---------------------------------------------------------------------
# Driver score calculation
# ---------------------------------------------------------------------

def get_driver_scores(cascade, source_identifier):
    ranking = source_identifier.rank_source_candidates(
        data=cascade,
        gnn_node_logits=None,
        method="rumor_centrality",
        top_k=10,
    )
    all_scores = source_identifier.predict_source_scores(
        data=cascade,
        gnn_node_logits=None,
        method="rumor_centrality",
    )
    return ranking, all_scores


# ---------------------------------------------------------------------
# Single dataset processor
# ---------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    checkpoint_path: Path,
    num_cascades: int = 10,
    num_snapshots: int = 4,
) -> Dict[str, Any]:
    """
    Process test cascades and load benchmark results for a single UPFD dataset.
    Extracts real observed timestamps for every cascade, phase, and user node.
    """
    print(f"\nProcessing dataset: {dataset_name.upper()}...")
    print(f"  Checkpoint: {checkpoint_path}")

    # Load UPFD dataset (BERT + multimodal features)
    _, _, test_ds, metadata = load_upfd_data(
        name=dataset_name,
        feature="bert",
        multimodal=True,
        use_synthetic_fallback=False,
    )

    in_dim = int(metadata["input_dim"])

    model = load_dashboard_model(
        checkpoint_path=checkpoint_path,
        in_dim=in_dim,
        num_snapshots=num_snapshots,
    )

    source_identifier = CascadeSourceIdentifier(hybrid_weight=0.7)
    anomaly_detector = BotNetworkIsolator(min_fanout=3, velocity_threshold=1.2)

    # Use the official timestamp-based builder matching the checkpoint
    temporal_builder = TemporalCascadeBuilder(
        num_snapshots=num_snapshots,
        add_temporal_features=True,
        require_complete_timestamps=True,
    )

    # 1. Load 10-epoch benchmark results
    bench_path = BENCHMARK_PATHS.get(dataset_name)
    benchmark_data = None
    if bench_path and bench_path.exists():
        try:
            with open(bench_path, "r", encoding="utf-8") as f:
                benchmark_data = json.load(f)
        except Exception as e:
            print(f"  [Warning] Failed to load benchmark from {bench_path}: {e}")

    # 2. Load extended training results
    ext_path = EXTENDED_PATHS.get(dataset_name)
    extended_data = None
    if ext_path and ext_path.exists():
        try:
            with open(ext_path, "r", encoding="utf-8") as f:
                extended_data = json.load(f)
        except Exception as e:
            print(f"  [Warning] Failed to load extended metrics from {ext_path}: {e}")

    # 3. Load saved driver analysis results
    driver_analysis_cache = {}
    driver_path = DRIVER_ANALYSIS_PATHS.get(dataset_name)
    if driver_path and driver_path.exists():
        try:
            with open(driver_path, "r", encoding="utf-8") as f:
                da = json.load(f)
                for item in da.get("results", []):
                    driver_analysis_cache[item.get("cascade_index")] = item
            print(f"  Loaded {len(driver_analysis_cache)} cached driver analysis records.")
        except Exception as e:
            print(f"  [Warning] Failed to load driver analysis from {driver_path}: {e}")

    # 4. Load early-stage evaluation results
    early_stage_path = EARLY_STAGE_PATHS.get(dataset_name)
    early_stage_data = None
    if early_stage_path and early_stage_path.exists():
        try:
            with open(early_stage_path, "r", encoding="utf-8") as f:
                early_stage_data = json.load(f)
            print(f"  Loaded early-stage evaluation results from {early_stage_path.name}.")
        except Exception as e:
            print(f"  [Warning] Failed to load early-stage metrics from {early_stage_path}: {e}")

    # 4. Process test cascades
    cascades = []
    total_test = len(test_ds)
    sample_count = total_test if num_cascades <= 0 else min(num_cascades, total_test)

    print(f"  Building {sample_count} test cascade records (total available: {total_test})...")

    for cascade_id in range(sample_count):
        cascade = test_ds[cascade_id]
        true_label = "FAKE" if int(cascade.y.item()) == 1 else "REAL"
        num_nodes = int(cascade.x.size(0))

        graph = build_networkx_tree(cascade.edge_index, num_nodes)

        # Structural driver candidates
        cached_record = driver_analysis_cache.get(cascade_id)
        if cached_record and "driver_candidates" in cached_record:
            cached_candidates = cached_record["driver_candidates"]
            driver_candidates = [
                {
                    "rank": r + 1,
                    "node": int(c["node"]),
                    "score": float(c["score"]),
                    "role": "Structural Propagation-Driver Candidate",
                }
                for r, c in enumerate(cached_candidates)
            ]
            driver_scores_dict = {
                int(c["node"]): float(c["score"]) for c in cached_candidates
            }
            # Fill remaining node scores
            all_scores = source_identifier.predict_source_scores(
                data=cascade,
                gnn_node_logits=None,
                method="rumor_centrality",
            )
            for n, s in all_scores.items():
                if int(n) not in driver_scores_dict:
                    driver_scores_dict[int(n)] = float(s)
        else:
            driver_ranking, all_driver_scores = get_driver_scores(
                cascade, source_identifier
            )
            driver_scores_dict = {int(k): float(v) for k, v in all_driver_scores.items()}
            driver_candidates = [
                {
                    "rank": r + 1,
                    "node": int(node),
                    "score": float(score),
                    "role": "Structural Propagation-Driver Candidate",
                }
                for r, (node, score) in enumerate(driver_ranking[:5])
            ]

        # Coordination / anomaly isolation
        anomaly_result = anomaly_detector.isolate_bot_rings(
            cascade, graph, threshold=0.45
        )
        anomaly_scores = {
            int(k): float(v)
            for k, v in anomaly_result["anomaly_scores"].items()
        }
        suspicious_nodes = [
            int(node) for node in anomaly_result["suspicious_nodes"]
        ]

        amplifier_candidates = sorted(
            [
                {
                    "node": int(node),
                    "score": float(anomaly_scores.get(int(node), 0.0)),
                    "fanout": int(graph.out_degree(node)) if node in graph else 0,
                    "role": "Heuristic Propagation Amplifier",
                }
                for node in suspicious_nodes
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:10]

        # Topology metrics
        cascade_metrics = compute_cascade_metrics(
            cascade.edge_index, num_nodes
        )

        # Overall cascade observed time span
        raw_user_times = (
            cascade.node_timestamps[1:]
            if hasattr(cascade, "node_timestamps") and cascade.node_timestamps.numel() > 1
            else None
        )
        if raw_user_times is not None:
            valid_user_times = raw_user_times[torch.isfinite(raw_user_times)]
            if valid_user_times.numel() > 0:
                min_time = float(valid_user_times.min().item())
                max_time = float(valid_user_times.max().item())
                min_dt = format_timestamp_utc(min_time)
                max_dt = format_timestamp_utc(max_time)
                duration_s = max(0.0, max_time - min_time)
            else:
                min_time, max_time, min_dt, max_dt, duration_s = None, None, None, None, 0.0
        else:
            min_time, max_time, min_dt, max_dt, duration_s = None, None, None, None, 0.0

        # Build true timestamp snapshots
        snapshots = temporal_builder.build_snapshots(cascade)

        phases = []
        for phase_idx, snapshot in enumerate(snapshots):
            active_mask = snapshot.active_mask
            active_count = int(active_mask.sum().item())

            edges = build_edge_records(snapshot.edge_index, active_mask)
            node_records = build_node_records(
                cascade=cascade,
                graph=graph,
                driver_scores=driver_scores_dict,
                anomaly_scores=anomaly_scores,
                suspicious_nodes=suspicious_nodes,
                active_mask=active_mask,
                driver_nodes=[c["node"] for c in driver_candidates],
            )

            prediction = predict_at_phase(
                model=model,
                snapshots=snapshots,
                phase_index=phase_idx,
            )

            # Chronological quantile cutoff
            cutoff_val = (
                float(snapshot.snapshot_cutoff_timestamp.item())
                if hasattr(snapshot, "snapshot_cutoff_timestamp")
                else None
            )
            cutoff_dt = format_timestamp_utc(cutoff_val)

            # Find latest user timestamp actually included in this snapshot
            active_node_ids = set(torch.where(active_mask)[0].tolist())
            active_user_ts = [
                float(cascade.node_timestamps[n].item())
                for n in active_node_ids
                if n > 0
                and hasattr(cascade, "node_timestamps")
                and n < cascade.node_timestamps.numel()
                and np.isfinite(cascade.node_timestamps[n].item())
            ]
            if active_user_ts:
                latest_observed_ts = max(active_user_ts)
                latest_observed_dt = format_timestamp_utc(latest_observed_ts)
            else:
                latest_observed_ts = cutoff_val
                latest_observed_dt = cutoff_dt

            quantile_pct = int(round(float((phase_idx + 1) / num_snapshots * 100)))

            elapsed_fraction = (
                float(snapshot.elapsed_fraction.item())
                if hasattr(snapshot, "elapsed_fraction")
                else float((phase_idx + 1) / num_snapshots)
            )

            phases.append({
                "index": phase_idx + 1,
                "label": f"Phase {phase_idx + 1} ({quantile_pct}% Quantile)",
                "quantile_pct": quantile_pct,
                "fraction": float((phase_idx + 1) / num_snapshots),
                "active_nodes": active_count,
                "cutoff_timestamp": cutoff_val,
                "cutoff_datetime_utc": cutoff_dt,
                "latest_observed_timestamp": latest_observed_ts,
                "latest_observed_datetime_utc": latest_observed_dt,
                "elapsed_fraction": elapsed_fraction,
                "prediction": prediction,
                "nodes": node_records,
                "edges": edges,
            })

        cascades.append({
            "id": cascade_id,
            "actual_label": true_label,
            "num_nodes": num_nodes,
            "max_depth": float(cascade_metrics.get("max_depth", 0)),
            "structural_virality": float(
                cascade_metrics.get("structural_virality", 0)
            ),
            "num_anomalous_amplifiers": len(suspicious_nodes),
            "earliest_timestamp": min_time,
            "earliest_datetime_utc": min_dt,
            "latest_timestamp": max_time,
            "latest_datetime_utc": max_dt,
            "duration_seconds": duration_s,
            "duration_human": format_duration(duration_s),
            "driver_candidates": driver_candidates,
            "amplifier_candidates": amplifier_candidates,
            "phases": phases,
        })

    print(f"  Done: processed {len(cascades)} cascades for {dataset_name}.")

    return {
        "dataset_key": dataset_name,
        "name": (
            "UPFD PolitiFact" if dataset_name == "politifact" else "UPFD GossipCop"
        ),
        "total_test_cascades": total_test,
        "loaded_cascades": len(cascades),
        "input_dim": in_dim,
        "benchmark": benchmark_data,
        "extended_results": extended_data,
        "early_stage": early_stage_data,
        "cascades": cascades,
    }


# ---------------------------------------------------------------------
# Dashboard data generator
# ---------------------------------------------------------------------

def generate_dashboard_data(
    dataset_choice: str = "all",
    num_cascades: int = 10,
    num_snapshots: int = 4,
    politifact_checkpoint: Optional[Path] = None,
    gossipcop_checkpoint: Optional[Path] = None,
) -> Path:
    """
    Generate research dashboard data for Politifact, GossipCop, or both.
    """
    print("\n" + "=" * 60)
    print("  GENERATING MULTI-DATASET DASHBOARD DATA")
    print("=" * 60)

    ckpts = {
        "politifact": politifact_checkpoint or DEFAULT_CHECKPOINTS["politifact"],
        "gossipcop": gossipcop_checkpoint or DEFAULT_CHECKPOINTS["gossipcop"],
    }

    datasets_to_process = (
        ["politifact", "gossipcop"]
        if dataset_choice == "all"
        else [dataset_choice]
    )

    processed_datasets = {}
    for ds_name in datasets_to_process:
        ckpt_path = ckpts[ds_name]
        processed_datasets[ds_name] = process_dataset(
            dataset_name=ds_name,
            checkpoint_path=ckpt_path,
            num_cascades=num_cascades,
            num_snapshots=num_snapshots,
        )

    # Active dataset
    primary_dataset = (
        "politifact" if "politifact" in processed_datasets else datasets_to_process[0]
    )

    payload = {
        "project": {
            "name": (
                "Spatio-Temporal Graph Neural Networks for Misinformation Cascade Tracking"
            ),
            "supported_datasets": list(processed_datasets.keys()),
            "active_dataset": primary_dataset,
            "temporal_mode": (
                "Timestamp-based chronological progression (quantile snapshots)"
            ),
            "num_snapshots": num_snapshots,
            "research_notes": {
                "source_localization": (
                    "Structural rumor-centrality approximations only. "
                    "UPFD does not provide verified human patient-zero labels."
                ),
                "anomaly_isolation": (
                    "Heuristic coordination and fanout metrics. "
                    "Not verified ground-truth bot detections."
                ),
                "temporal_quantiles": (
                    "Chronological quantile cutoffs represent observed diffusion progress, "
                    "not fixed-duration real-time streaming windows."
                ),
            },
        },
        "datasets": processed_datasets,
        "early_stage": {
            ds_name: processed_datasets[ds_name].get("early_stage")
            for ds_name in processed_datasets
        },
        # Backward compatibility for direct top-level access:
        "benchmark": processed_datasets[primary_dataset]["benchmark"],
        "cascades": processed_datasets[primary_dataset]["cascades"],
    }

    output_path = PROJECT_ROOT / "web" / "dashboard_data.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            to_json_value(payload),
            f,
            indent=2,
            allow_nan=False,
        )

    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nDashboard data written successfully:")
    print(f"  Path: {output_path}")
    print(f"  Size: {file_size_kb:.1f} KB")
    print(f"  Datasets included: {list(processed_datasets.keys())}")

    return output_path


# ---------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------

def run_server(
    port: int = 8080,
    open_browser: bool = False,
    dataset: str = "all",
    num_cascades: int = 10,
    politifact_checkpoint: Optional[str] = None,
    gossipcop_checkpoint: Optional[str] = None,
):
    pol_ckpt = (
        Path(politifact_checkpoint).resolve()
        if politifact_checkpoint
        else DEFAULT_CHECKPOINTS["politifact"]
    )
    gos_ckpt = (
        Path(gossipcop_checkpoint).resolve()
        if gossipcop_checkpoint
        else DEFAULT_CHECKPOINTS["gossipcop"]
    )

    # Generate fresh dashboard data backing both datasets
    generate_dashboard_data(
        dataset_choice=dataset,
        num_cascades=num_cascades,
        num_snapshots=4,
        politifact_checkpoint=pol_ckpt,
        gossipcop_checkpoint=gos_ckpt,
    )

    web_dir = PROJECT_ROOT / "web"
    if not web_dir.exists():
        raise FileNotFoundError(f"Web directory not found: {web_dir}")

    os.chdir(web_dir)

    selected_port = find_available_port(port)
    if selected_port != port:
        print(f"[Notice] Port {port} is occupied. Using available port {selected_port}.")

    handler = http.server.SimpleHTTPRequestHandler

    with http.server.ThreadingHTTPServer(("", selected_port), handler) as httpd:
        url = f"http://localhost:{selected_port}"
        print("\n" + "=" * 60)
        print("  SPATIO-TEMPORAL GNN RESEARCH DASHBOARD")
        print(f"  Serving at: {url}")
        print("  Datasets active: PolitiFact and GossipCop")
        print("  Press Ctrl+C to terminate.")
        print("=" * 60 + "\n")

        if open_browser:
            webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server terminated cleanly.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start interactive STGNN cascade research dashboard with Politifact & GossipCop support."
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server")
    parser.add_argument("--open", action="store_true", help="Open default browser")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "politifact", "gossipcop"],
        help="Dataset to load into dashboard ('all' includes both).",
    )
    parser.add_argument(
        "--num_cascades",
        type=int,
        default=10,
        help="Number of test cascades to include per dataset (default: 10). Set <= 0 for all.",
    )
    parser.add_argument(
        "--all_cascades",
        action="store_true",
        help="Load all available test cascades for the selected datasets.",
    )
    parser.add_argument(
        "--politifact_checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINTS["politifact"]),
        help="Path to Politifact STGNN checkpoint.",
    )
    parser.add_argument(
        "--gossipcop_checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINTS["gossipcop"]),
        help="Path to GossipCop STGNN checkpoint.",
    )

    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Generate dashboard_data.json and exit without starting the web server.",
    )

    args = parser.parse_args()

    cascade_count = 0 if args.all_cascades else args.num_cascades

    if args.generate_only:
        generate_dashboard_data(
            dataset_choice=args.dataset,
            num_cascades=cascade_count,
            num_snapshots=4,
            politifact_checkpoint=Path(args.politifact_checkpoint),
            gossipcop_checkpoint=Path(args.gossipcop_checkpoint),
        )
        print("Data generation complete. Exiting (--generate_only).")
        sys.exit(0)

    run_server(
        port=args.port,
        open_browser=args.open,
        dataset=args.dataset,
        num_cascades=cascade_count,
        politifact_checkpoint=args.politifact_checkpoint,
        gossipcop_checkpoint=args.gossipcop_checkpoint,
    )