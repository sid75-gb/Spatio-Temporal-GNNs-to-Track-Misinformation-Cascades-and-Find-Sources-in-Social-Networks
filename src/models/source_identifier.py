"""
Propagation-driver and source-candidate identification.

Important UPFD semantic rule:

    node 0 = news/root node
    node 1..N-1 = propagation users

Therefore node 0 is NOT treated as a human source candidate.

This module supports:

1. GNN node scoring
2. Rumor-centrality scoring
3. Hybrid scoring
4. Earliest-observed-propagator identification
5. Hit@K / MRR / hop-distance evaluation

The earliest observed propagator should NOT be described as the
historical "true patient zero".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from ..algorithms.rumor_centrality import (
    compute_rumor_centrality_all_nodes,
)
from ..data.cascade_features import (
    build_networkx_tree,
)


class CascadeSourceIdentifier:
    """
    Scores propagation users as possible cascade drivers.

    The news/root node (0) is excluded from user-source ranking.
    """

    def __init__(
        self,
        hybrid_weight: float = 0.7,
    ):
        if not 0.0 <= hybrid_weight <= 1.0:
            raise ValueError(
                "hybrid_weight must be between 0 and 1."
            )

        self.hybrid_weight = float(
            hybrid_weight
        )

    # -----------------------------------------------------------------
    # Candidate users
    # -----------------------------------------------------------------

    @staticmethod
    def _candidate_nodes(
        num_nodes: int,
    ) -> List[int]:
        """
        Return user nodes only.

        UPFD:
            node 0 = news/root
            node 1..N-1 = users
        """
        if num_nodes <= 1:
            return []

        return list(
            range(1, num_nodes)
        )

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        scores: Dict[int, float],
        candidates: List[int],
    ) -> Dict[int, float]:
        """
        Normalize candidate scores so they sum to 1.

        This prevents combining quantities with incompatible raw scales.
        """

        if not candidates:
            return {}

        values = np.array(
            [
                max(
                    0.0,
                    float(
                        scores.get(
                            node,
                            0.0,
                        )
                    ),
                )
                for node in candidates
            ],
            dtype=np.float64,
        )

        total = float(
            values.sum()
        )

        if (
            not np.isfinite(total)
            or total <= 0.0
        ):
            uniform = (
                1.0 / len(candidates)
            )

            return {
                node: uniform
                for node in candidates
            }

        values /= total

        return {
            node: float(
                values[i]
            )
            for i, node in enumerate(
                candidates
            )
        }

    # -----------------------------------------------------------------
    # Earliest observed propagator
    # -----------------------------------------------------------------

    @staticmethod
    def find_earliest_observed_propagators(
        data: Data,
    ) -> List[int]:
        """
        Find the user node(s) with the earliest observed timestamp.

        Root/news node 0 is excluded.

        Returns all users tied for the minimum timestamp.
        """

        timestamps = getattr(
            data,
            "node_timestamps",
            None,
        )

        if timestamps is None:
            raise ValueError(
                "Graph has no node_timestamps. "
                "Load official UPFD timestamp metadata first."
            )

        timestamps = (
            timestamps.detach()
            .cpu()
            .float()
            .view(-1)
        )

        if timestamps.numel() <= 1:
            return []

        user_times = timestamps[
            1:
        ]

        valid_mask = torch.isfinite(
            user_times
        )

        if not bool(
            valid_mask.any()
        ):
            return []

        valid_indices = torch.where(
            valid_mask
        )[0] + 1

        valid_times = user_times[
            valid_mask
        ]

        min_time = float(
            valid_times.min().item()
        )

        earliest_nodes = [
            int(node)
            for node, timestamp in zip(
                valid_indices.tolist(),
                valid_times.tolist(),
            )
            if float(timestamp) == min_time
        ]

        return earliest_nodes

    def infer_earliest_observed_propagator(
        self,
        data: Data,
    ) -> Optional[int]:
        """
        Return one earliest observed propagator.

        When several users share exactly the same timestamp, the smallest
        local node ID is selected deterministically.
        """

        candidates = (
            self.find_earliest_observed_propagators(
                data
            )
        )

        if not candidates:
            return None

        return min(
            candidates
        )

    # -----------------------------------------------------------------
    # Score calculation
    # -----------------------------------------------------------------

    def predict_source_scores(
        self,
        data: Data,
        gnn_node_logits: Optional[torch.Tensor] = None,
        method: str = "hybrid",
    ) -> Dict[int, float]:
        """
        Compute propagation-driver scores.

        Supported methods:

            "gnn"
            "rumor_centrality"
            "hybrid"

        Only user nodes (1..N-1) are returned.
        """

        method = method.lower()

        if method not in {
            "gnn",
            "rumor_centrality",
            "hybrid",
        }:
            raise ValueError(
                f"Unknown source scoring method '{method}'. "
                "Use 'gnn', 'rumor_centrality', or 'hybrid'."
            )

        num_nodes = int(
            data.num_nodes
            if data.num_nodes is not None
            else data.x.size(0)
        )

        candidates = (
            self._candidate_nodes(
                num_nodes
            )
        )

        if not candidates:
            return {}

        graph = build_networkx_tree(
            data.edge_index,
            num_nodes,
        )

        # -------------------------------------------------------------
        # Rumor-centrality / structural score
        # -------------------------------------------------------------
        raw_rumor_scores = (
            compute_rumor_centrality_all_nodes(
                graph,
                root_hint=0,
            )
        )

        rumor_scores = (
            self._normalize_scores(
                raw_rumor_scores,
                candidates,
            )
        )

        # -------------------------------------------------------------
        # GNN learned scores
        # -------------------------------------------------------------
        if gnn_node_logits is None:
            gnn_scores = {
                node: 1.0 / len(candidates)
                for node in candidates
            }

        else:
            logits = (
                gnn_node_logits
                .detach()
                .view(-1)
                .cpu()
            )

            if logits.numel() < num_nodes:
                raise ValueError(
                    "gnn_node_logits contains fewer values "
                    "than the number of cascade nodes."
                )

            candidate_logits = logits[
                torch.tensor(
                    candidates,
                    dtype=torch.long,
                )
            ]

            probabilities = F.softmax(
                candidate_logits,
                dim=0,
            ).numpy()

            gnn_scores = {
                node: float(
                    probabilities[i]
                )
                for i, node in enumerate(
                    candidates
                )
            }

        # -------------------------------------------------------------
        # Return selected scoring method
        # -------------------------------------------------------------
        if method == "gnn":
            return gnn_scores

        if method == "rumor_centrality":
            return rumor_scores

        # -------------------------------------------------------------
        # Hybrid normalized consensus
        # -------------------------------------------------------------
        w = self.hybrid_weight

        hybrid_scores = {
            node: float(
                w * gnn_scores[node]
                + (1.0 - w)
                * rumor_scores[node]
            )
            for node in candidates
        }

        return self._normalize_scores(
            hybrid_scores,
            candidates,
        )

    # -----------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------

    def rank_source_candidates(
        self,
        data: Data,
        gnn_node_logits: Optional[torch.Tensor] = None,
        method: str = "hybrid",
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Return the highest-scoring propagation-driver candidates.
        """

        scores = self.predict_source_scores(
            data=data,
            gnn_node_logits=gnn_node_logits,
            method=method,
        )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[
            :max(1, int(top_k))
        ]

    # -----------------------------------------------------------------
    # Source evaluation
    # -----------------------------------------------------------------

    def evaluate_cascade_source(
        self,
        data: Data,
        gnn_node_logits: Optional[torch.Tensor] = None,
        true_source: Optional[int] = None,
        method: str = "hybrid",
        top_k_list: Optional[List[int]] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Evaluate propagation-driver localization.

        If true_source is None, the earliest observed propagator from the
        official timestamp metadata is used.

        IMPORTANT:
            node 0 is never considered a valid user source target.
        """

        if top_k_list is None:
            top_k_list = [1, 3, 5]

        num_nodes = int(
            data.num_nodes
            if data.num_nodes is not None
            else data.x.size(0)
        )

        candidates = (
            self._candidate_nodes(
                num_nodes
            )
        )

        if not candidates:
            raise ValueError(
                "Cascade has no user nodes to evaluate."
            )

        # -------------------------------------------------------------
        # Determine target
        # -------------------------------------------------------------
        if true_source is None:
            inferred_source = (
                self.infer_earliest_observed_propagator(
                    data
                )
            )

            if inferred_source is None:
                raise ValueError(
                    "Cannot evaluate source localization because no "
                    "valid earliest-observed propagator can be inferred "
                    "from node timestamps."
                )

            true_source = inferred_source

        true_source = int(
            true_source
        )

        # Node 0 is the news/root node and cannot be a user source target.
        if true_source == 0:
            raise ValueError(
                "Invalid true_source=0. UPFD node 0 is the news/root node, "
                "not a human propagation-user source. "
                "Use an observed user node (>=1)."
            )

        if true_source not in candidates:
            raise ValueError(
                f"true_source={true_source} is not a valid user node."
            )

        # -------------------------------------------------------------
        # Rank all candidates
        # -------------------------------------------------------------
        ranked = self.rank_source_candidates(
            data=data,
            gnn_node_logits=gnn_node_logits,
            method=method,
            top_k=len(candidates),
        )

        ranked_nodes = [
            node
            for node, _ in ranked
        ]

        if true_source not in ranked_nodes:
            raise RuntimeError(
                "Internal source-ranking error: true source "
                "was not present among ranked user nodes."
            )

        true_rank = (
            ranked_nodes.index(
                true_source
            )
            + 1
        )

        reciprocal_rank = (
            1.0 / true_rank
        )

        # -------------------------------------------------------------
        # Hit@K
        # -------------------------------------------------------------
        results: Dict[
            str,
            Union[float, int],
        ] = {
            "valid_source_target": 1.0,
            "true_rank": int(
                true_rank
            ),
            "mrr": float(
                reciprocal_rank
            ),
            "true_source": int(
                true_source
            ),
        }

        for k in top_k_list:
            k = int(k)

            results[
                f"hit@{k}"
            ] = (
                1.0
                if true_rank <= k
                else 0.0
            )

        # -------------------------------------------------------------
        # Hop-distance error
        # -------------------------------------------------------------
        graph = build_networkx_tree(
            data.edge_index,
            num_nodes,
        )

        predicted_top1 = (
            ranked_nodes[0]
        )

        try:
            graph_undirected = (
                graph.to_undirected()
            )

            hop_error = (
                nx.shortest_path_length(
                    graph_undirected,
                    predicted_top1,
                    true_source,
                )
            )

        except Exception:
            hop_error = (
                0
                if predicted_top1 == true_source
                else float("nan")
            )

        results[
            "top1_predicted"
        ] = int(
            predicted_top1
        )

        results[
            "error_distance"
        ] = float(
            hop_error
        )

        return results