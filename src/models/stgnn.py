"""
Spatio-Temporal Graph Neural Network (STGNN) for misinformation
cascade classification and propagation-driver scoring.

Architecture:

    Node Features
         ↓
    Input Projection
         ↓
    Bidirectional GAT
         ↓
       GRU
         ↓
    Temporal Self-Attention
         ↓
    ┌───────────────┐
    │               │
    ↓               ↓
Graph Veracity   Node Driver
  Classifier      Scoring Head

Important:
The node scoring head produces learned node scores.
It does NOT by itself prove historical patient-zero identity.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GlobalAttention


# ---------------------------------------------------------------------
# Spatial layer
# ---------------------------------------------------------------------

class SpatialDiffusionLayer(nn.Module):
    """
    Direction-aware graph attention layer.

    Two separate message-passing directions are modeled:

        forward:
            parent → child

        reverse:
            child → parent

    We call this bidirectional structural message passing rather than
    claiming that the reverse direction represents human verification.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if num_heads < 1:
            raise ValueError(
                "num_heads must be >= 1."
            )

        head_dim = max(
            1,
            out_dim // num_heads,
        )

        gat_dim = (
            head_dim * num_heads
        )

        self.gat_forward = GATConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )

        self.gat_backward = GATConv(
            in_channels=in_dim,
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(
            gat_dim
        )

        self.proj = nn.Linear(
            gat_dim,
            out_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        if edge_index.numel() == 0:
            # No edges at this snapshot.
            return self.proj(x)

        # -------------------------------------------------------------
        # Parent → child
        # -------------------------------------------------------------
        h_forward = self.gat_forward(
            x,
            edge_index,
        )

        # -------------------------------------------------------------
        # Child → parent
        # -------------------------------------------------------------
        reverse_edge_index = torch.stack(
            [
                edge_index[1],
                edge_index[0],
            ],
            dim=0,
        )

        h_backward = self.gat_backward(
            x,
            reverse_edge_index,
        )

        # -------------------------------------------------------------
        # Fuse directional structural messages
        # -------------------------------------------------------------
        h_fused = (
            h_forward
            + h_backward
        )

        h_fused = self.norm(
            h_fused
        )

        return self.proj(
            h_fused
        )


# ---------------------------------------------------------------------
# STGNN
# ---------------------------------------------------------------------

class SpatioTemporalGNN(nn.Module):
    """
    Unified Spatio-Temporal GNN.

    Outputs:

        veracity_logits:
            [B, 2]

        source_scores:
            [N, 1]

        temporal_node_states:
            list containing node states for each snapshot.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_snapshots: int = 4,
        temporal_cell: str = "gru",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads}) for temporal attention."
            )

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_snapshots = num_snapshots
        self.temporal_cell_type = (
            temporal_cell.lower()
        )
        self.dropout = dropout

        # -------------------------------------------------------------
        # Input projection
        # -------------------------------------------------------------
        self.input_proj = nn.Linear(
            in_features,
            hidden_dim,
        )

        # -------------------------------------------------------------
        # Spatial encoder
        # -------------------------------------------------------------
        self.spatial_layer = (
            SpatialDiffusionLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        # -------------------------------------------------------------
        # Temporal encoder
        # -------------------------------------------------------------
        if self.temporal_cell_type == "lstm":
            self.temporal_cell = nn.LSTMCell(
                hidden_dim,
                hidden_dim,
            )
        else:
            self.temporal_cell = nn.GRUCell(
                hidden_dim,
                hidden_dim,
            )

            self.temporal_cell_type = "gru"

        # -------------------------------------------------------------
        # Temporal self-attention
        # -------------------------------------------------------------
        self.temporal_attn = (
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=dropout,
            )
        )

        # -------------------------------------------------------------
        # Graph attention pooling
        # -------------------------------------------------------------
        gate_nn = nn.Sequential(
            nn.Linear(
                hidden_dim,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                1,
            ),
        )

        self.graph_pool = GlobalAttention(
            gate_nn=gate_nn,
        )

        # -------------------------------------------------------------
        # Graph-level fake/real classifier
        # -------------------------------------------------------------
        self.veracity_classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

        # -------------------------------------------------------------
        # Node-level propagation-driver scoring
        # -------------------------------------------------------------
        self.source_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _get_active_mask(
        snapshot: Data,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a boolean active-node mask."""

        active_mask = getattr(
            snapshot,
            "active_mask",
            None,
        )

        if active_mask is None:
            return torch.ones(
                num_nodes,
                dtype=torch.bool,
                device=device,
            )

        active_mask = active_mask.to(
            device=device,
            dtype=torch.bool,
        ).view(-1)

        if active_mask.numel() != num_nodes:
            raise ValueError(
                "active_mask length does not match "
                "the number of nodes."
            )

        return active_mask

    @staticmethod
    def _get_batch_indices(
        first_snapshot: Data,
        num_nodes: int,
        device: torch.device,
        batch_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Determine graph membership for each node."""

        if batch_idx is not None:
            batch_idx = batch_idx.to(
                device=device,
                dtype=torch.long,
            )

            if batch_idx.numel() != num_nodes:
                raise ValueError(
                    "batch_idx length does not match "
                    "the number of nodes."
                )

            return batch_idx

        snapshot_batch = getattr(
            first_snapshot,
            "batch",
            None,
        )

        if snapshot_batch is not None:
            return snapshot_batch.to(
                device=device,
                dtype=torch.long,
            )

        return torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=device,
        )

    @staticmethod
    def _root_indices_from_batch(
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find the first node of each graph.

        UPFD's root is node 0 within each individual cascade.
        """

        num_graphs = int(
            batch_idx.max().item()
        ) + 1

        root_indices = []

        for graph_id in range(
            num_graphs
        ):
            indices = torch.where(
                batch_idx == graph_id
            )[0]

            if indices.numel() == 0:
                continue

            root_indices.append(
                indices[0]
            )

        if not root_indices:
            return torch.empty(
                (0,),
                dtype=torch.long,
                device=batch_idx.device,
            )

        return torch.stack(
            root_indices
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward_snapshots(
        self,
        snapshot_list: List[Data],
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[torch.Tensor],
    ]:
        """
        Process a temporal cascade sequence.

        Each snapshot should have:

            x
            edge_index
            active_mask
            node_timestamps
        """

        if not snapshot_list:
            raise ValueError(
                "snapshot_list must contain at least one snapshot."
            )

        T = len(
            snapshot_list
        )

        if (
            self.num_snapshots > 0
            and T != self.num_snapshots
        ):
            raise ValueError(
                f"Model expects {self.num_snapshots} snapshots, "
                f"but received {T}."
            )

        first_snapshot = snapshot_list[0]

        num_nodes = int(
            first_snapshot.x.size(0)
        )

        device = first_snapshot.x.device

        # -------------------------------------------------------------
        # Initial temporal states
        # -------------------------------------------------------------
        h_temporal = torch.zeros(
            num_nodes,
            self.hidden_dim,
            device=device,
        )

        if self.temporal_cell_type == "lstm":
            c_temporal = torch.zeros(
                num_nodes,
                self.hidden_dim,
                device=device,
            )

        snapshot_states: List[
            torch.Tensor
        ] = []

        active_history: List[
            torch.Tensor
        ] = []

        # -------------------------------------------------------------
        # Process temporal snapshots
        # -------------------------------------------------------------
        for snapshot in snapshot_list:

            x_t = snapshot.x.to(
                device
            )

            edge_index_t = (
                snapshot.edge_index.to(
                    device
                )
            )

            active_mask = (
                self._get_active_mask(
                    snapshot,
                    num_nodes,
                    device,
                )
            )

            # Record active-node history for temporal attention.
            active_history.append(
                active_mask
            )

            # ---------------------------------------------------------
            # Input projection
            # ---------------------------------------------------------
            x_proj = F.gelu(
                self.input_proj(
                    x_t
                )
            )

            # Prevent inactive nodes from contributing.
            x_proj = x_proj * active_mask.unsqueeze(
                -1
            ).float()

            # ---------------------------------------------------------
            # Spatial graph encoder
            # ---------------------------------------------------------
            h_spatial = (
                self.spatial_layer(
                    x_proj,
                    edge_index_t,
                )
            )

            h_spatial = F.dropout(
                h_spatial,
                p=self.dropout,
                training=self.training,
            )

            h_spatial = (
                h_spatial
                * active_mask.unsqueeze(
                    -1
                ).float()
            )

            # ---------------------------------------------------------
            # Temporal recurrent update
            # ---------------------------------------------------------
            if self.temporal_cell_type == "lstm":

                candidate_h, candidate_c = (
                    self.temporal_cell(
                        h_spatial,
                        (
                            h_temporal,
                            c_temporal,
                        ),
                    )
                )

                h_temporal = torch.where(
                    active_mask.unsqueeze(-1),
                    candidate_h,
                    torch.zeros_like(
                        candidate_h
                    ),
                )

                c_temporal = torch.where(
                    active_mask.unsqueeze(-1),
                    candidate_c,
                    torch.zeros_like(
                        candidate_c
                    ),
                )

            else:

                candidate_h = (
                    self.temporal_cell(
                        h_spatial,
                        h_temporal,
                    )
                )

                h_temporal = torch.where(
                    active_mask.unsqueeze(-1),
                    candidate_h,
                    torch.zeros_like(
                        candidate_h
                    ),
                )

            snapshot_states.append(
                h_temporal
            )

        # -------------------------------------------------------------
        # [N, T, D]
        # -------------------------------------------------------------
        trajectory = torch.stack(
            snapshot_states,
            dim=1,
        )

        active_history_tensor = torch.stack(
            active_history,
            dim=1,
        ).bool()

        # MultiheadAttention uses True = ignore/padding.
        temporal_padding_mask = (
            ~active_history_tensor
        )

        # Prevent NaNs when a node is inactive at every time step.
        # Give such rows one valid attention key; the final inactive-node
        # mask below will still zero out their final embeddings.
        all_masked_rows = temporal_padding_mask.all(dim=1)

        if all_masked_rows.any():
            temporal_padding_mask[all_masked_rows, 0] = False

        temporal_attn_out, _ = (
            self.temporal_attn(
                trajectory,
                trajectory,
                trajectory,
                key_padding_mask=temporal_padding_mask,
            )
        )

        # Final state: last valid snapshot + attention representation.
        final_node_embeds = (
            temporal_attn_out[:, -1, :]
            + h_temporal
        )

        # Ensure inactive nodes cannot leak into the final embedding.
        final_active_mask = (
            active_history_tensor[:, -1]
        )

        final_node_embeds = (
            final_node_embeds
            * final_active_mask.unsqueeze(-1).float()
        )

        # -------------------------------------------------------------
        # Graph membership
        # -------------------------------------------------------------
        batch_idx = (
            self._get_batch_indices(
                first_snapshot,
                num_nodes,
                device,
                batch_idx,
            )
        )

        # -------------------------------------------------------------
        # Graph-level pooling
        # -------------------------------------------------------------
        graph_pooled = self.graph_pool(
            final_node_embeds,
            batch_idx,
        )

        # -------------------------------------------------------------
        # Root/news node embedding
        # -------------------------------------------------------------
        root_indices = (
            self._root_indices_from_batch(
                batch_idx
            )
        )

        if root_indices.numel() == 0:
            root_embeds = torch.zeros(
                graph_pooled.size(0),
                self.hidden_dim,
                device=device,
            )
        else:
            root_embeds = final_node_embeds[
                root_indices
            ]

        # -------------------------------------------------------------
        # Veracity prediction
        # -------------------------------------------------------------
        cascade_summary = torch.cat(
            [
                graph_pooled,
                root_embeds,
            ],
            dim=-1,
        )

        veracity_logits = (
            self.veracity_classifier(
                cascade_summary
            )
        )

        # -------------------------------------------------------------
        # Node-level driver/source score
        # -------------------------------------------------------------
        source_scores = (
            self.source_head(
                final_node_embeds
            )
        )

        return (
            veracity_logits,
            source_scores,
            snapshot_states,
        )

    def forward(
        self,
        snapshot_list: List[Data],
        batch_idx: Optional[torch.Tensor] = None,
    ):
        """Alias for forward_snapshots."""
        return self.forward_snapshots(
            snapshot_list,
            batch_idx=batch_idx,
        )