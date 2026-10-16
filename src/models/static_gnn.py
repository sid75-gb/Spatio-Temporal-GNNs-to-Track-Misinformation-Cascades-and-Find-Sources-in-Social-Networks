"""
Static Graph Neural Network baseline architectures for cascade veracity classification.
Implements GCN, GAT, and GraphSAGE models operating on static propagation trees.
"""
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, GlobalAttention


class StaticCascadeGNN(nn.Module):
    """
    Static GNN baseline for misinformation cascade classification.
    Aggregates structural interactions across the static propagation tree.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "gat",
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = "attention",
    ):
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling

        # Feature projection layer
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Build GNN Conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim

            if self.gnn_type == "gat":
                # Ensure out_dim divisible by num_heads
                head_dim = max(1, out_dim // num_heads)
                conv = GATConv(in_dim, head_dim, heads=num_heads, concat=True, dropout=dropout)
                norm = nn.LayerNorm(head_dim * num_heads)
            elif self.gnn_type == "gcn":
                conv = GCNConv(in_dim, out_dim)
                norm = nn.LayerNorm(out_dim)
            elif self.gnn_type == "sage":
                conv = SAGEConv(in_dim, out_dim)
                norm = nn.LayerNorm(out_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")

            self.convs.append(conv)
            self.norms.append(norm)

        # Graph-level pooling
        if pooling == "attention":
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))
            self.pool = GlobalAttention(gate_nn=gate_nn)
            pool_out_dim = hidden_dim
        else:
            self.pool = None
            pool_out_dim = hidden_dim

        # Final classification MLP
        # Fuses pooled graph representation with root node representation
        self.classifier = nn.Sequential(
            nn.Linear(pool_out_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass:
        Args:
            batch: PyG Batch or Data containing cascade graphs.
        Returns:
            Logits of shape [batch_size, num_classes].
        """
        x, edge_index = batch.x, batch.edge_index
        batch_idx = batch.batch if hasattr(batch, "batch") and batch.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial projection
        h = F.relu(self.input_proj(x))

        # Graph convolutions with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if h.shape == h_res.shape:
                h = h + h_res

        # Graph-level pooling
        if self.pooling_type == "attention":
            g_pool = self.pool(h, batch_idx)
        elif self.pooling_type == "max":
            g_pool = global_max_pool(h, batch_idx)
        else:
            g_pool = global_mean_pool(h, batch_idx)

        # Extract root node embedding for each cascade in batch
        if hasattr(batch, "ptr") and batch.ptr is not None:
            root_indices = batch.ptr[:-1]
        else:
            root_indices = torch.tensor([0], device=h.device)
        root_embeds = h[root_indices]

        # Concatenate graph summary with root node state
        cascade_repr = torch.cat([g_pool, root_embeds], dim=-1)

        return self.classifier(cascade_repr)
