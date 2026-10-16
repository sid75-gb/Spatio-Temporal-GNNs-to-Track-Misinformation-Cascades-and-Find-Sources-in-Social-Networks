"""
NLP Baseline Classifier: Content-only veracity detection using root post text representations.
Demonstrates traditional linguistic-only detection without topological cascade knowledge.
"""
from typing import Union
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


class NLPContentClassifier(nn.Module):
    """
    Classifies misinformation solely from the root news content embedding (node 0 in cascade).
    Architectural structure: MLP with Dropout, LayerNorm, and GELU activations.
    """

    def __init__(self, in_features: int = 768, hidden_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def extract_root_features(self, batch: Union[Data, Batch]) -> torch.Tensor:
        """Extracts the feature vector of the root post (node 0) for each graph in the batch."""
        if isinstance(batch, Batch):
            # For PyG Batches, each graph's root node is the first node of that graph
            # ptr indicates start of each graph's nodes
            ptr = batch.ptr[:-1]
            root_feats = batch.x[ptr]
        else:
            root_feats = batch.x[0:1]

        # If multimodal features are present (e.g. 778-d), take text portion
        if root_feats.size(-1) > self.in_features:
            root_feats = root_feats[:, :self.in_features]
        elif root_feats.size(-1) < self.in_features:
            # Pad if needed
            pad_size = self.in_features - root_feats.size(-1)
            root_feats = torch.cat([root_feats, torch.zeros(root_feats.size(0), pad_size, device=root_feats.device)], dim=-1)

        return root_feats

    def forward(self, batch: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass:
        Args:
            batch: PyG Batch or Data containing cascade graphs.
        Returns:
            Logits of shape [batch_size, num_classes].
        """
        root_feats = self.extract_root_features(batch)
        return self.classifier(root_feats)
