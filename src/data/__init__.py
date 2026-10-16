"""Data processing, temporal cascade generation, and feature extraction."""
from .cascade_features import (
    build_networkx_tree,
    compute_structural_virality,
    compute_tree_levels,
    compute_cascade_metrics,
)
from .temporal_builder import TemporalCascadeBuilder, collate_temporal_snapshots
from .dataset_loader import (
    MultimodalUPFDDataset,
    SyntheticCascadeDataset,
    load_upfd_data,
    get_dataloaders,
)

__all__ = [
    "build_networkx_tree",
    "compute_structural_virality",
    "compute_tree_levels",
    "compute_cascade_metrics",
    "TemporalCascadeBuilder",
    "collate_temporal_snapshots",
    "MultimodalUPFDDataset",
    "SyntheticCascadeDataset",
    "load_upfd_data",
    "get_dataloaders",
]
