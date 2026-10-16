"""
Standalone Evaluation of Earliest-Observed-Retweeter Proxy.

MSc Data Science Research Experiment:
"Spatio-Temporal Graph Neural Networks for Misinformation Cascade Tracking
and Source Node Identification in Social Networks"

SCIENTIFIC DISCLAIMER:
    This experiment evaluates identification of the EARLIEST-OBSERVED RETWEETER
    in crawled UPFD cascades.
    The target is a timestamp-derived observational crawl proxy.
    It does NOT represent the true misinformation author, original publisher,
    human patient zero, or a verified source account.
    UPFD contains no verified ground-truth author or source-account annotations.

METHODS EVALUATED:
    1. Random User Baseline (Reproducible seed 42 + Analytical Expectation)
    2. Out-Degree Centrality (Retweet fan-out in directed cascade tree)
    3. Rumor Centrality (Shah & Zaman 2011 structural message passing)
    4. Earliest-Timestamp Oracle (Upper-bound sanity check; not a discovery method)

NOTE:
    The untrained model.source_head is intentionally excluded from this evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.rumor_centrality import compute_rumor_centrality_all_nodes
from src.data.cascade_features import build_networkx_tree
from src.data.dataset_loader import load_upfd_data


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_eligible_users(cascade) -> Tuple[List[int], List[int], Dict[int, float]]:
    """
    Extract eligible user nodes and the target earliest-observed retweeter(s).

    Rules:
        - Node 0 (news article claim) is strictly excluded.
        - Users must have finite, valid timestamps (> 0).

    Returns:
        candidate_users: List of user node indices (1..N-1) with valid timestamps.
        target_nodes: List of user node indices tied for the earliest timestamp.
        timestamps: Dict mapping candidate node index -> timestamp float.
    """
    num_nodes = cascade.x.size(0)
    if num_nodes <= 1:
        return [], [], {}

    raw_ts = getattr(cascade, "node_timestamps", None)
    if raw_ts is None:
        return [], [], {}

    ts_tensor = raw_ts.detach().cpu().numpy().flatten()
    if ts_tensor.size <= 1:
        return [], [], {}

    user_timestamps: Dict[int, float] = {}
    for node_idx in range(1, num_nodes):
        if node_idx < len(ts_tensor):
            t_val = ts_tensor[node_idx]
            if np.isfinite(t_val) and t_val > 0:
                user_timestamps[node_idx] = float(t_val)

    if not user_timestamps:
        return [], [], {}

    candidates = sorted(user_timestamps.keys())
    min_time = min(user_timestamps.values())
    targets = [node for node, t in user_timestamps.items() if t == min_time]

    return candidates, targets, user_timestamps


def rank_random(candidates: List[int], rng: random.Random) -> List[int]:
    """Randomly shuffle candidate users."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


def rank_out_degree(
    G: nx.DiGraph,
    candidates: List[int],
    rng: random.Random,
) -> List[int]:
    """
    Rank candidate users by out-degree in the directed tree (retweet fan-out).
    Ties are shuffled randomly using rng to avoid arbitrary index bias.
    """
    degrees = {u: G.out_degree(u) if u in G else 0 for u in candidates}
    # Group by degree to shuffle ties fairly
    deg_groups: Dict[int, List[int]] = {}
    for u, deg in degrees.items():
        deg_groups.setdefault(deg, []).append(u)

    ranked: List[int] = []
    for deg in sorted(deg_groups.keys(), reverse=True):
        group = list(deg_groups[deg])
        rng.shuffle(group)
        ranked.extend(group)

    return ranked


def rank_rumor_centrality(
    G: nx.DiGraph,
    candidates: List[int],
    rng: random.Random,
) -> List[int]:
    """
    Rank candidate users by Shah-Zaman Rumor Centrality.
    Ties are shuffled randomly using rng to avoid arbitrary index bias.
    """
    scores = compute_rumor_centrality_all_nodes(G, root_hint=0)
    user_scores = {u: scores.get(u, 0.0) for u in candidates}

    score_groups: Dict[float, List[int]] = {}
    for u, sc in user_scores.items():
        # Round slightly to group float ties robustly
        rounded = round(sc, 8)
        score_groups.setdefault(rounded, []).append(u)

    ranked: List[int] = []
    for sc in sorted(score_groups.keys(), reverse=True):
        group = list(score_groups[sc])
        rng.shuffle(group)
        ranked.extend(group)

    return ranked


def rank_oracle(
    candidates: List[int],
    timestamps: Dict[int, float],
    rng: random.Random,
) -> List[int]:
    """
    Sanity-check Oracle: Sort ascending by timestamp.
    Ties are shuffled randomly using rng.
    """
    time_groups: Dict[float, List[int]] = {}
    for u in candidates:
        t = timestamps[u]
        time_groups.setdefault(t, []).append(u)

    ranked: List[int] = []
    for t in sorted(time_groups.keys()):
        group = list(time_groups[t])
        rng.shuffle(group)
        ranked.extend(group)

    return ranked


def compute_hop_distance(G: nx.DiGraph, u: int, v: int) -> float:
    """Compute shortest path distance in undirected tree."""
    if u == v:
        return 0.0
    try:
        undir = G.to_undirected()
        return float(nx.shortest_path_length(undir, u, v))
    except Exception:
        return float("nan")


def score_ranking(
    ranked: List[int],
    targets: Set[int],
    G: nx.DiGraph,
) -> Dict[str, float]:
    """
    Score a single ranking against the set of target nodes.

    Conventions:
        - Multi-target hit: At least one target in top-k.
        - Best reciprocal rank: 1 / min(rank(t) for t in targets).
        - Average reciprocal rank: mean(1 / rank(t) for t in targets).
        - Hop distance error: Shortest path from ranked[0] to closest target.
    """
    # 1-indexed ranks
    rank_dict = {node: idx + 1 for idx, node in enumerate(ranked)}
    target_ranks = [rank_dict[t] for t in targets if t in rank_dict]

    if not target_ranks:
        return {
            "hit@1": 0.0,
            "hit@3": 0.0,
            "hit@5": 0.0,
            "mrr": 0.0,
            "mrr_avg": 0.0,
            "hop_error": float("nan"),
        }

    best_rank = min(target_ranks)
    mrr_best = 1.0 / best_rank
    mrr_avg = float(np.mean([1.0 / r for r in target_ranks]))

    top1 = ranked[0]
    hop_errors = [compute_hop_distance(G, top1, t) for t in targets]
    valid_hops = [h for h in hop_errors if np.isfinite(h)]
    min_hop = min(valid_hops) if valid_hops else float("nan")

    return {
        "hit@1": 1.0 if best_rank <= 1 else 0.0,
        "hit@3": 1.0 if best_rank <= 3 else 0.0,
        "hit@5": 1.0 if best_rank <= 5 else 0.0,
        "mrr": mrr_best,
        "mrr_avg": mrr_avg,
        "hop_error": min_hop,
    }


def evaluate_dataset(
    dataset_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate source proxy on official held-out test split."""
    print(f"\n" + "=" * 65)
    print(f"  EVALUATING EARLIEST-RETWEETER PROXY: {dataset_name.upper()}")
    print("=" * 65)

    set_seed(seed)
    rng = random.Random(seed)

    _, _, test_ds, meta = load_upfd_data(
        name=dataset_name,
        feature="bert",
        multimodal=False,
    )

    total_test_cascades = len(test_ds)
    print(f"Total test cascades in split: {total_test_cascades}")

    eligible_cascades: List[Dict[str, Any]] = []
    skipped_cascades: List[Dict[str, Any]] = []

    # Prepare common set of eligible cascades
    for idx in range(total_test_cascades):
        cascade = test_ds[idx]
        candidates, targets, timestamps = extract_eligible_users(cascade)

        if not candidates or len(candidates) < 1:
            skipped_cascades.append({
                "cascade_idx": idx,
                "reason": "No user nodes with valid timestamps (num_nodes <= 1 or invalid timestamps)",
            })
            continue

        if not targets:
            skipped_cascades.append({
                "cascade_idx": idx,
                "reason": "Could not identify minimum timestamp among user nodes",
            })
            continue

        G = build_networkx_tree(cascade.edge_index, cascade.x.size(0))

        eligible_cascades.append({
            "cascade_idx": idx,
            "num_total_nodes": cascade.x.size(0),
            "num_candidates": len(candidates),
            "candidates": candidates,
            "targets": targets,
            "is_tied": len(targets) > 1,
            "num_targets": len(targets),
            "timestamps": timestamps,
            "graph": G,
            "actual_label": "FAKE" if cascade.y.item() == 1 else "REAL",
        })

    num_eligible = len(eligible_cascades)
    num_skipped = len(skipped_cascades)
    num_tied = sum(1 for c in eligible_cascades if c["is_tied"])

    print(f"Eligible cascades: {num_eligible} of {total_test_cascades}")
    print(f"Skipped cascades: {num_skipped}")
    print(f"Cascades with tied earliest timestamps: {num_tied} ({num_tied / max(1, num_eligible) * 100:.1f}%)")

    # Evaluate methods
    method_results: Dict[str, List[Dict[str, float]]] = {
        "random": [],
        "out_degree": [],
        "rumor_centrality": [],
        "oracle_timestamp": [],
    }

    # Analytical expectation for random baseline
    analytical_random_hit1 = []
    analytical_random_hit3 = []
    analytical_random_hit5 = []
    analytical_random_mrr = []

    for item in eligible_cascades:
        candidates = item["candidates"]
        targets_set = set(item["targets"])
        G = item["graph"]
        M = len(candidates)
        T_cnt = len(targets_set)

        # Analytical random expectation
        analytical_random_hit1.append(min(1.0, T_cnt * 1.0 / M))
        analytical_random_hit3.append(min(1.0, T_cnt * 3.0 / M))
        analytical_random_hit5.append(min(1.0, T_cnt * 5.0 / M))
        # Expected rank for random draw
        exp_mrr_c = float(np.mean([1.0 / r for r in range(1, M + 1)]))
        analytical_random_mrr.append(exp_mrr_c)

        # 1. Random ranking
        ranked_rand = rank_random(candidates, rng)
        method_results["random"].append(score_ranking(ranked_rand, targets_set, G))

        # 2. Out-degree ranking
        ranked_deg = rank_out_degree(G, candidates, rng)
        method_results["out_degree"].append(score_ranking(ranked_deg, targets_set, G))

        # 3. Rumor centrality ranking
        ranked_rc = rank_rumor_centrality(G, candidates, rng)
        method_results["rumor_centrality"].append(score_ranking(ranked_rc, targets_set, G))

        # 4. Oracle (Sanity check)
        ranked_orc = rank_oracle(candidates, item["timestamps"], rng)
        method_results["oracle_timestamp"].append(score_ranking(ranked_orc, targets_set, G))

    def aggregate_metrics(res_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not res_list:
            return {}
        return {
            "hit@1": float(np.mean([r["hit@1"] for r in res_list])),
            "hit@3": float(np.mean([r["hit@3"] for r in res_list])),
            "hit@5": float(np.mean([r["hit@5"] for r in res_list])),
            "mrr": float(np.mean([r["mrr"] for r in res_list])),
            "mrr_avg_convention": float(np.mean([r["mrr_avg"] for r in res_list])),
            "mean_hop_distance": float(np.nanmean([r["hop_error"] for r in res_list])),
        }

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "proxy_definition": {
            "target": "earliest_observed_retweeter",
            "candidate_set": "user_nodes_with_valid_timestamps (1..N-1)",
            "excluded_nodes": ["node_0_news_article"],
            "tie_handling": "Multi-target convention (hit if top-K contains any tied earliest retweeter; best rank used for MRR)",
            "alternative_tie_handling": "Average reciprocal rank across all tied targets included as mrr_avg_convention",
            "scientific_disclaimer": (
                "The target is an observational timestamp crawl proxy. "
                "It does NOT represent verified misinformation authors, original publishers, "
                "or ground-truth patient zero."
            ),
        },
        "cascade_counts": {
            "total_test_cascades": total_test_cascades,
            "evaluated_eligible_cascades": num_eligible,
            "skipped_cascades": num_skipped,
            "tied_earliest_cascades": num_tied,
            "tied_pct": float(num_tied / max(1, num_eligible) * 100),
            "skipped_reasons": skipped_cascades,
        },
        "baselines_and_methods": {
            "random_baseline": aggregate_metrics(method_results["random"]),
            "random_analytical_expectation": {
                "hit@1": float(np.mean(analytical_random_hit1)),
                "hit@3": float(np.mean(analytical_random_hit3)),
                "hit@5": float(np.mean(analytical_random_hit5)),
                "mean_mrr_approx": float(np.mean(analytical_random_mrr)),
            },
            "out_degree_centrality": aggregate_metrics(method_results["out_degree"]),
            "rumor_centrality": aggregate_metrics(method_results["rumor_centrality"]),
        },
        "sanity_check_oracle": {
            "note": "Upper-bound verification check sorting directly by timestamp; NOT a discovery method.",
            "metrics": aggregate_metrics(method_results["oracle_timestamp"]),
        },
    }

    # Print summary table
    print("\n--- RESULTS SUMMARY ---")
    print(f"{'Method':<30} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR':<8} | {'Hop Dist':<8}")
    print("-" * 80)

    for m_name, m_key in [
        ("Random Baseline (Seed 42)", "random_baseline"),
        ("Out-Degree Centrality", "out_degree_centrality"),
        ("Rumor Centrality (Shah-Zaman)", "rumor_centrality"),
    ]:
        m_vals = summary["baselines_and_methods"][m_key]
        print(
            f"{m_name:<30} | "
            f"{m_vals['hit@1']*100:>6.2f}% | "
            f"{m_vals['hit@3']*100:>6.2f}% | "
            f"{m_vals['hit@5']*100:>6.2f}% | "
            f"{m_vals['mrr']:>7.4f} | "
            f"{m_vals['mean_hop_distance']:>7.2f}"
        )

    orc_vals = summary["sanity_check_oracle"]["metrics"]
    print("-" * 80)
    print(
        f"{'[Oracle] Earliest Timestamp':<30} | "
        f"{orc_vals['hit@1']*100:>6.2f}% | "
        f"{orc_vals['hit@3']*100:>6.2f}% | "
        f"{orc_vals['hit@5']*100:>6.2f}% | "
        f"{orc_vals['mrr']:>7.4f} | "
        f"{orc_vals['mean_hop_distance']:>7.2f}"
    )
    print("-" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate earliest-observed-retweeter proxy on UPFD benchmark."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "politifact", "gossipcop"],
        help="Dataset to evaluate (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save JSON results",
    )
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["politifact", "gossipcop"] if args.dataset == "all" else [args.dataset]

    all_summaries = {}
    for ds in datasets:
        summary = evaluate_dataset(ds, seed=args.seed)
        out_path = out_dir / f"source_proxy_{ds}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved evaluation results to: {out_path}")
        all_summaries[ds] = summary

    print("\n" + "=" * 65)
    print("  ALL PROXY EVALUATIONS COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
