"""
Temporal cascade builder for dynamic UPFD propagation graphs.

Unlike the previous implementation, this module DOES NOT invent timestamps
from graph depth or node index.

It uses the official UPFD node timestamps attached by dataset_loader.py.

Each snapshot contains only propagation information that was observable
by that timestamp cutoff. The temporal progression feature represents the
known observation fraction (e.g., 0.25, 0.50, 0.75, 1.00) rather than
elapsed wall-clock time normalized by the unobserved future cascade duration,
strictly preventing future-information leakage.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from .cascade_features import (
    build_networkx_tree,
)


class TemporalCascadeBuilder:
    """
    Construct temporal graph snapshots from real UPFD timestamps.

    For each cascade:

        G(t1) -> G(t2) -> ... -> G(T)

    where each snapshot contains only nodes whose observed timestamp is
    <= the snapshot cutoff.

    Root node 0 is always active because it represents the news item and
    does not normally have a timestamp in UPFD.

    The temporal progression feature reflects the predefined observation
    fraction for the current snapshot, strictly avoiding any dependence
    on future timestamps or total cascade duration.
    """

    def __init__(
        self,
        num_snapshots: int = 4,
        add_temporal_features: bool = True,
        observation_fractions: Optional[Sequence[float]] = None,
        require_complete_timestamps: bool = True,
    ):
        self.num_snapshots = max(
            2,
            int(num_snapshots),
        )

        self.add_temporal_features = (
            add_temporal_features
        )

        self.require_complete_timestamps = (
            require_complete_timestamps
        )

        if observation_fractions is None:
            # Research-friendly defaults.
            #
            # Existing 4-phase dashboard:
            # 25%, 50%, 75%, 100%
            #
            # For 5 snapshots we use:
            # 10%, 25%, 50%, 75%, 100%
            if self.num_snapshots == 4:
                fractions = [
                    0.25,
                    0.50,
                    0.75,
                    1.00,
                ]
            elif self.num_snapshots == 5:
                fractions = [
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    1.00,
                ]
            else:
                fractions = np.linspace(
                    1.0 / self.num_snapshots,
                    1.0,
                    self.num_snapshots,
                ).tolist()

            self.observation_fractions = [
                float(x) for x in fractions
            ]

        else:
            fractions = [
                float(x)
                for x in observation_fractions
            ]

            if len(fractions) < 2:
                raise ValueError(
                    "At least two observation fractions are required."
                )

            if any(
                x <= 0.0 or x > 1.0
                for x in fractions
            ):
                raise ValueError(
                    "Observation fractions must be in (0, 1]."
                )

            if any(
                fractions[i] >= fractions[i + 1]
                for i in range(len(fractions) - 1)
            ):
                raise ValueError(
                    "Observation fractions must be strictly increasing."
                )

            self.observation_fractions = fractions
            self.num_snapshots = len(fractions)

    # -----------------------------------------------------------------
    # Timestamp validation
    # -----------------------------------------------------------------

    def _get_node_timestamps(
        self,
        data: Data,
    ) -> torch.Tensor:
        """Get and validate real node timestamps."""

        timestamps = getattr(
            data,
            "node_timestamps",
            None,
        )

        if timestamps is None:
            raise ValueError(
                "TemporalCascadeBuilder requires "
                "'data.node_timestamps'. "
                "The graph does not contain UPFD timestamp metadata."
            )

        timestamps = timestamps.to(dtype=torch.float64).view(-1)

        if timestamps.numel() != data.x.size(0):
            raise ValueError(
                "Timestamp/node count mismatch: "
                f"{timestamps.numel()} timestamps for "
                f"{data.x.size(0)} nodes."
            )

        if data.num_nodes <= 1:
            raise ValueError(
                "Temporal modeling requires at least one propagating "
                "user node in addition to the news root."
            )

        # Root timestamp is normally absent and represented as NaN.
        user_timestamps = timestamps[1:]

        valid_mask = torch.isfinite(
            user_timestamps
        )

        if self.require_complete_timestamps and not bool(
            valid_mask.all()
        ):
            missing_nodes = (
                torch.where(~valid_mask)[0] + 1
            ).tolist()

            preview = missing_nodes[:10]

            raise ValueError(
                "Incomplete UPFD timestamp coverage. "
                f"Missing timestamps for {len(missing_nodes)} user nodes "
                f"(example local node IDs: {preview}). "
                "Do not fabricate timestamps. Check the official "
                "*_id_time_mapping.pkl files and node alignment."
            )

        if not bool(valid_mask.any()):
            raise ValueError(
                "No valid user timestamps were found for this cascade."
            )

        return timestamps

    # -----------------------------------------------------------------
    # Snapshot fractions
    # -----------------------------------------------------------------

    def _select_cutoff_timestamp(
        self,
        user_timestamps: torch.Tensor,
        fraction: float,
    ) -> float:
        """
        Select a chronological cutoff containing approximately the requested
        fraction of observed users.
        """

        valid_times = user_timestamps[
            torch.isfinite(user_timestamps)
        ]

        if valid_times.numel() == 0:
            raise ValueError(
                "Cannot create a temporal snapshot without timestamps."
            )

        sorted_times, _ = torch.sort(
            valid_times
        )

        n = sorted_times.numel()

        # ceil ensures the requested fraction is represented by at least
        # one observed propagation event.
        k = int(
            np.ceil(
                fraction * n
            )
        )

        k = max(
            1,
            min(k, n),
        )

        return float(
            sorted_times[k - 1].item()
        )

    # -----------------------------------------------------------------
    # Snapshot structural features
    # -----------------------------------------------------------------

    def _build_snapshot_features(
        self,
        edge_index: torch.Tensor,
        active_mask: torch.Tensor,
        num_nodes: int,
        observation_fraction: float = 1.0,
        elapsed_fraction: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute structural-temporal features using ONLY the currently
        observed subgraph.

        Features:
            1. current normalized cascade depth
            2. current normalized out-degree
            3. observation fraction (known requested fraction, e.g. 0.25, 0.50,
               0.75, 1.00; strictly independent of future cascade duration)
            4. active-node indicator (1.0 for active nodes, 0.0 for inactive nodes)
        """
        if elapsed_fraction is not None:
            progression_val = float(elapsed_fraction)
        else:
            progression_val = float(observation_fraction)

        graph = build_networkx_tree(
            edge_index,
            num_nodes,
        )

        active_nodes = torch.where(
            active_mask
        )[0].tolist()

        if len(active_nodes) == 0:
            active_nodes = [0]

        active_graph = graph.subgraph(
            active_nodes
        ).copy()

        # Ensure root exists.
        if 0 not in active_graph:
            active_graph.add_node(0)

        # -------------------------------------------------------------
        # Current depth
        # -------------------------------------------------------------
        depth_dict = {}

        try:
            depth_dict = dict(
                __import__(
                    "networkx"
                ).single_source_shortest_path_length(
                    active_graph,
                    0,
                )
            )
        except Exception:
            depth_dict = {0: 0}

        max_depth = max(
            depth_dict.values(),
            default=0,
        )

        # -------------------------------------------------------------
        # Current out-degree
        # -------------------------------------------------------------
        out_degree_dict = dict(
            active_graph.out_degree()
        )

        max_out_degree = max(
            out_degree_dict.values(),
            default=0,
        )

        temporal_features = torch.zeros(
            num_nodes,
            4,
            dtype=torch.float32,
        )

        for node in active_nodes:
            current_depth = float(
                depth_dict.get(
                    node,
                    0,
                )
            )

            current_out_degree = float(
                out_degree_dict.get(
                    node,
                    0,
                )
            )

            normalized_depth = (
                current_depth / max_depth
                if max_depth > 0
                else 0.0
            )

            normalized_out_degree = (
                current_out_degree / max_out_degree
                if max_out_degree > 0
                else 0.0
            )

            temporal_features[node] = torch.tensor(
                [
                    normalized_depth,
                    normalized_out_degree,
                    progression_val,
                    1.0,
                ],
                dtype=torch.float32,
            )

        return temporal_features

    # -----------------------------------------------------------------
    # Main snapshot construction
    # -----------------------------------------------------------------

    def build_snapshots(
        self,
        data: Data,
    ) -> List[Data]:
        """
        Build temporal snapshots from real UPFD timestamps.
        """

        num_nodes = int(data.x.size(0))

        edge_index = data.edge_index

        timestamps = self._get_node_timestamps(
            data
        )

        user_timestamps = timestamps[1:]

        snapshots: List[Data] = []

        for step_idx, fraction in enumerate(
            self.observation_fractions
        ):
            cutoff = self._select_cutoff_timestamp(
                user_timestamps,
                fraction,
            )

            # ---------------------------------------------------------
            # Active node mask
            # ---------------------------------------------------------
            active_mask = torch.zeros(
                num_nodes,
                dtype=torch.bool,
            )

            # Root/news item always exists.
            active_mask[0] = True

            # Real observed users up to cutoff.
            user_active = (
                torch.isfinite(user_timestamps)
                & (user_timestamps <= cutoff)
            )

            active_mask[1:] = user_active

            # ---------------------------------------------------------
            # Active edges only
            # ---------------------------------------------------------
            if edge_index.numel() > 0:
                src = edge_index[0]
                dst = edge_index[1]

                edge_mask = (
                    active_mask[src]
                    & active_mask[dst]
                )

                snapshot_edge_index = (
                    edge_index[:, edge_mask]
                )
            else:
                snapshot_edge_index = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )

            # ---------------------------------------------------------
            # Temporal progression / observation fraction
            # ---------------------------------------------------------
            # The progression feature represents the predefined snapshot
            # observation fraction (e.g. 0.25, 0.50, 0.75, 1.00), strictly
            # avoiding any dependence on the total duration or final arrival
            # timestamp of future, unobserved cascade participants.
            observation_fraction = float(fraction)

            # ---------------------------------------------------------
            # Current-only structural features
            # ---------------------------------------------------------
            temporal_features = (
                self._build_snapshot_features(
                    edge_index=snapshot_edge_index,
                    active_mask=active_mask,
                    num_nodes=num_nodes,
                    observation_fraction=observation_fraction,
                )
            )

            # ---------------------------------------------------------
            # Node input features
            # ---------------------------------------------------------
            node_x = data.x.float().clone()

            if self.add_temporal_features:
                node_x = torch.cat(
                    [
                        node_x,
                        temporal_features,
                    ],
                    dim=-1,
                )

            # Prevent inactive future users from contributing feature
            # information before they actually appear.
            inactive_mask = ~active_mask
            node_x[inactive_mask] = 0.0

            snapshot = Data(
                x=node_x,
                edge_index=snapshot_edge_index,
                y=data.y.clone()
                if data.y is not None
                else None,
                active_mask=active_mask,
                snapshot_step=torch.tensor(
                    step_idx,
                    dtype=torch.long,
                ),
                snapshot_fraction=torch.tensor(
                    fraction,
                    dtype=torch.float32,
                ),
                snapshot_cutoff_timestamp=torch.tensor(
                    cutoff,
                    dtype=torch.float64,
                ),
                elapsed_fraction=torch.tensor(
                    observation_fraction,
                    dtype=torch.float32,
                ),
                observation_fraction=torch.tensor(
                    observation_fraction,
                    dtype=torch.float32,
                ),
                node_timestamps=timestamps.clone(),
                timestamp_mask=torch.isfinite(
                    timestamps
                ),
                num_nodes=num_nodes,
            )

            if hasattr(data, "global_node_ids"):
                snapshot.global_node_ids = (
                    data.global_node_ids.clone()
                )

            if hasattr(data, "upfd_graph_id"):
                snapshot.upfd_graph_id = (
                    data.upfd_graph_id
                )

            snapshots.append(
                snapshot
            )

        return snapshots


# ---------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------

def collate_temporal_snapshots(
    batch_graphs: List[Data],
    builder: TemporalCascadeBuilder,
) -> List[List[Data]]:
    """
    Convert:

        [graph_1, graph_2, ..., graph_B]

    into:

        [
            [graph_1@T1, graph_2@T1, ...],
            [graph_1@T2, graph_2@T2, ...],
            ...
        ]
    """

    all_snapshots = [
        builder.build_snapshots(graph)
        for graph in batch_graphs
    ]

    num_steps = builder.num_snapshots

    step_batches: List[List[Data]] = []

    for t in range(num_steps):
        step_batches.append(
            [
                graph_snapshots[t]
                for graph_snapshots in all_snapshots
            ]
        )

    return step_batches