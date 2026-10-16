"""
Dataset loader for User Preference-aware Fake News Detection (UPFD).

This module:
1. Loads the official UPFD benchmark through PyTorch Geometric.
2. Fuses UPFD BERT/user-history features with profile features.
3. Loads the official UPFD node-to-timestamp mappings.
4. Attaches real node timestamps to each graph.
5. Uses training-set-only statistics for profile normalization.
6. Never silently falls back to synthetic data during research runs.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _coerce_timestamp(value) -> Optional[float]:
    """Convert a timestamp-like value to float, or return None."""
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def _find_file(
    filename: str,
    roots: List[Path],
) -> Optional[Path]:
    """
    Search a small set of likely dataset locations.

    The official manually downloaded UPFD layout stores auxiliary mapping
    files under the repository's data directory, while the raw graph files
    live under data/{dataset}/raw.
    """
    checked = set()

    for root in roots:
        root = root.resolve()

        candidates = [
            root / filename,
            root / "data" / filename,
        ]

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in checked:
                continue

            checked.add(candidate)

            if candidate.exists():
                return candidate

    # Controlled recursive search.
    for root in roots:
        root = root.resolve()

        if not root.exists():
            continue

        try:
            for candidate in root.rglob(filename):
                candidate = candidate.resolve()

                if candidate in checked:
                    continue

                checked.add(candidate)

                if candidate.is_file():
                    return candidate

        except (OSError, PermissionError):
            continue

    return None


# ---------------------------------------------------------------------
# UPFD timestamp resolver
# ---------------------------------------------------------------------

class UPFDTimestampResolver:
    """
    Resolves local graph-node indices to the official UPFD timestamps.

    UPFD stores:
        - node_graph_id.npy
        - train_idx.npy / val_idx.npy / test_idx.npy
        - <prefix>_id_time_mapping.pkl

    node_graph_id.npy gives the original global graph assignment for every
    global node. The split index files map a local split item to the original
    graph index.

    The time mapping uses the original/global node IDs.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
    ):
        self.root = Path(root)
        self.name = name.lower()
        self.split = split

        if self.name == "politifact":
            self.prefix = "pol"
        elif self.name == "gossipcop":
            self.prefix = "gos"
        else:
            raise ValueError(
                f"Unsupported UPFD dataset '{name}'. "
                f"Expected 'politifact' or 'gossipcop'."
            )

        self.raw_dir = self.root / self.name / "raw"

        # The official auxiliary mapping files are normally under data/.
        search_roots = [
            self.root,
            self.root.parent,
            self.root.parent.parent,
            Path.cwd(),
            Path.cwd() / "data",
        ]

        self.time_mapping_path = _find_file(
            f"{self.prefix}_id_time_mapping.pkl",
            search_roots,
        )

        self.node_graph_id_path = self.raw_dir / "node_graph_id.npy"
        self.split_idx_path = self.raw_dir / f"{split}_idx.npy"

        self.time_mapping: Dict[int, float] = {}
        self.node_graph_id: Optional[np.ndarray] = None
        self.split_graph_ids: Optional[np.ndarray] = None

        self.available = False

        self._load()

    def _load(self) -> None:
        """Load all timestamp-related metadata."""
        if self.time_mapping_path is None:
            return

        if not self.node_graph_id_path.exists():
            return

        if not self.split_idx_path.exists():
            return

        try:
            with open(self.time_mapping_path, "rb") as f:
                mapping = pickle.load(f)

            if not isinstance(mapping, dict):
                raise ValueError(
                    f"Expected timestamp mapping to be a dict, "
                    f"got {type(mapping).__name__}."
                )

            parsed_mapping: Dict[int, float] = {}

            for node_id, timestamp in mapping.items():
                try:
                    node_id_int = int(node_id)
                except (TypeError, ValueError):
                    continue

                timestamp_float = _coerce_timestamp(timestamp)

                if timestamp_float is not None:
                    parsed_mapping[node_id_int] = timestamp_float

            self.time_mapping = parsed_mapping

            self.node_graph_id = np.asarray(
                np.load(self.node_graph_id_path, allow_pickle=False)
            ).reshape(-1)

            if np.any(np.diff(self.node_graph_id) < 0):
                raise ValueError(
                    "node_graph_id.npy must be ordered by graph ID "
                    "for efficient timestamp resolution."
                )

            self.split_graph_ids = np.asarray(
                np.load(self.split_idx_path, allow_pickle=False)
            ).reshape(-1)

            if self.node_graph_id.size == 0:
                raise ValueError("node_graph_id.npy is empty.")

            if self.split_graph_ids.size == 0:
                raise ValueError(
                    f"{self.split_idx_path.name} is empty."
                )

            self.available = True

        except Exception as exc:
            print(
                f"Warning: Could not initialize UPFD timestamps for "
                f"{self.name}-{self.split}: {exc}"
            )

            self.time_mapping = {}
            self.node_graph_id = None
            self.split_graph_ids = None
            self.available = False

    def resolve(
        self,
        local_graph_index: int,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Resolve timestamps for one graph.

        Returns:
            node_timestamps:
                [N] float tensor.
                Root/news timestamp is normally NaN because the original
                UPFD/FakeNewsNet data does not provide it.

            global_node_ids:
                [N] integer tensor containing the original global node IDs.

            global_graph_id:
                Original graph index in the UPFD raw dataset.
        """
        if not self.available:
            raise FileNotFoundError(
                "\nUPFD timestamp metadata is not available.\n\n"
                f"Expected timestamp mapping similar to:\n"
                f"  {self.prefix}_id_time_mapping.pkl\n\n"
                f"Expected raw file:\n"
                f"  {self.node_graph_id_path}\n\n"
                f"Expected split file:\n"
                f"  {self.split_idx_path}\n\n"
                "The official UPFD repository provides the "
                "*_id_time_mapping.pkl files and node_graph_id/split "
                "metadata. Download/copy those files before running "
                "the temporal STGNN experiment."
            )

        assert self.node_graph_id is not None
        assert self.split_graph_ids is not None

        if local_graph_index < 0 or local_graph_index >= len(self.split_graph_ids):
            raise IndexError(
                f"Graph index {local_graph_index} is outside the "
                f"{self.split} split of size {len(self.split_graph_ids)}."
            )

        global_graph_id = int(self.split_graph_ids[local_graph_index])

        # Original/global node IDs belonging to this graph.
        # The graph IDs are sorted, so binary search avoids
        # scanning the entire dataset for every cascade.
        start = int(
            np.searchsorted(
                self.node_graph_id,
                global_graph_id,
                side="left",
            )
        )

        end = int(
            np.searchsorted(
                self.node_graph_id,
                global_graph_id,
                side="right",
            )
        )

        global_node_positions = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        if len(global_node_positions) != num_nodes:
            raise ValueError(
                f"UPFD node alignment error for {self.name}-{self.split}, "
                f"graph {global_graph_id}: "
                f"PyG graph has {num_nodes} nodes, but node_graph_id.npy "
                f"contains {len(global_node_positions)} nodes for that graph."
            )

        node_timestamps = np.full(
            num_nodes,
            np.nan,
            dtype=np.float64,
        )

        for local_idx, global_node_id in enumerate(global_node_positions):
            timestamp = self.time_mapping.get(
                int(global_node_id),
                None,
            )

            if timestamp is not None:
                node_timestamps[local_idx] = float(timestamp)

        global_node_ids = torch.tensor(
            global_node_positions,
            dtype=torch.long,
        )

        timestamp_tensor = torch.tensor(
            node_timestamps,
            dtype=torch.float64,
        )

        return (
            timestamp_tensor,
            global_node_ids,
            global_graph_id,
        )


# ---------------------------------------------------------------------
# Synthetic dataset - debugging only
# ---------------------------------------------------------------------

class SyntheticCascadeDataset(Dataset):
    """
    Synthetic diffusion cascade dataset.

    IMPORTANT:
    This dataset exists only for development/unit testing.

    It must not be used for reported UPFD research results.
    """

    def __init__(
        self,
        num_graphs: int = 50,
        feature_dim: int = 768,
        seed: int = 42,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.graphs: List[Data] = []

        for i in range(num_graphs):
            # Label: 0 = Real, 1 = Fake.
            y = torch.tensor([i % 2], dtype=torch.long)
            is_fake = y.item() == 1

            num_nodes = int(
                np.random.randint(25, 80)
                if is_fake
                else np.random.randint(15, 50)
            )

            x = torch.randn(
                num_nodes,
                feature_dim,
            )

            src_list = []
            dst_list = []

            for child in range(1, num_nodes):
                if is_fake and np.random.rand() < 0.4:
                    parent = 0
                else:
                    parent = int(np.random.randint(0, child))

                src_list.append(parent)
                dst_list.append(child)

            edge_index = torch.tensor(
                [src_list, dst_list],
                dtype=torch.long,
            )

            node_timestamps = torch.full(
                (num_nodes,),
                float("nan"),
                dtype=torch.float32,
            )

            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                node_timestamps=node_timestamps,
                timestamp_mask=torch.zeros(
                    num_nodes,
                    dtype=torch.bool,
                ),
                synthetic_data=torch.tensor(
                    True,
                    dtype=torch.bool,
                ),
            )

            self.graphs.append(data)

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx].clone()


# ---------------------------------------------------------------------
# UPFD multimodal dataset
# ---------------------------------------------------------------------

class MultimodalUPFDDataset(Dataset):
    """
    UPFD dataset with:

        base feature
        +
        profile feature

    The original feature representations are preserved separately:

        data.bert_x
        data.profile_x

    In addition, real UPFD timestamps are attached when available.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
        feature: str = "bert",
        multimodal: bool = True,
    ):
        super().__init__()

        self.root = root
        self.name = name
        self.split = split
        self.feature = feature
        self.multimodal = multimodal

        self.primary_ds = UPFD(
            root=root,
            name=name,
            feature=feature,
            split=split,
        )

        self.profile_ds: Optional[UPFD] = None

        if multimodal and feature != "profile":
            try:
                self.profile_ds = UPFD(
                    root=root,
                    name=name,
                    feature="profile",
                    split=split,
                )
            except Exception as exc:
                print(
                    "Warning: Could not load UPFD profile features: "
                    f"{exc}"
                )
                self.profile_ds = None

        self.timestamp_resolver = UPFDTimestampResolver(
            root=root,
            name=name,
            split=split,
        )

        self.profile_mean: Optional[torch.Tensor] = None
        self.profile_std: Optional[torch.Tensor] = None

    def set_profile_stats(
        self,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> None:
        """Set training-derived profile normalization statistics."""
        self.profile_mean = mean
        self.profile_std = std

    def timestamps_available(self) -> bool:
        """Return whether official UPFD timestamp metadata was found."""
        return self.timestamp_resolver.available

    def _get_timestamp_data(
        self,
        idx: int,
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        """
        Attach official timestamps.

        Missing root timestamp is expected.
        """
        if not self.timestamp_resolver.available:
            timestamps = torch.full(
                (num_nodes,),
                float("nan"),
                dtype=torch.float32,
            )

            global_node_ids = torch.arange(
                num_nodes,
                dtype=torch.long,
            )

            return timestamps, global_node_ids, None

        return self.timestamp_resolver.resolve(
            local_graph_index=idx,
            num_nodes=num_nodes,
        )

    def len(self) -> int:
        return len(self.primary_ds)

    def get(self, idx: int) -> Data:
        data = self.primary_ds[idx].clone()

        if data.x is None:
            raise ValueError(
                f"UPFD graph {idx} has no node features."
            )

        base_x = data.x.float().clone()

        # Preserve base representation explicitly.
        if self.feature == "bert":
            data.bert_x = base_x.clone()
        else:
            data.base_feature_x = base_x.clone()

        # -------------------------------------------------------------
        # Profile features
        # -------------------------------------------------------------
        if self.profile_ds is not None:
            profile_data = self.profile_ds[idx]

            if (
                profile_data.x is not None
                and profile_data.x.size(0) == base_x.size(0)
            ):
                profile_x = profile_data.x.float().clone()

                # Training-set-derived normalization only.
                if (
                    self.profile_mean is not None
                    and self.profile_std is not None
                ):
                    profile_x = (
                        profile_x - self.profile_mean
                    ) / self.profile_std

                data.profile_x = profile_x.clone()

                fused_x = torch.cat(
                    [base_x, profile_x],
                    dim=-1,
                )

                data.x = fused_x

        # -------------------------------------------------------------
        # Real UPFD timestamp metadata
        # -------------------------------------------------------------
        timestamps, global_node_ids, global_graph_id = (
            self._get_timestamp_data(
                idx=idx,
                num_nodes=data.x.size(0),
            )
        )

        data.node_timestamps = timestamps

        # Root/news node normally has NaN timestamp.
        data.timestamp_mask = torch.isfinite(timestamps)

        # Preserve original/global node identifiers.
        data.global_node_ids = global_node_ids

        if global_graph_id is not None:
            data.upfd_graph_id = int(global_graph_id)

        data.synthetic_data = torch.tensor(
            False,
            dtype=torch.bool,
        )

        return data


# ---------------------------------------------------------------------
# Training-only profile statistics
# ---------------------------------------------------------------------

def _compute_training_profile_stats(
    dataset: MultimodalUPFDDataset,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute profile normalization statistics using TRAINING data only.

    This avoids calculating normalization independently for each graph and
    avoids validation/test leakage.
    """
    if dataset.profile_ds is None:
        return None, None

    feature_dim = None

    sum_x = None
    sum_x2 = None
    count = 0

    for idx in range(len(dataset.profile_ds)):
        profile_data = dataset.profile_ds[idx]

        if profile_data.x is None:
            continue

        x = profile_data.x.float()

        if x.dim() != 2:
            continue

        if feature_dim is None:
            feature_dim = x.size(1)
            sum_x = torch.zeros(
                feature_dim,
                dtype=torch.float64,
            )
            sum_x2 = torch.zeros(
                feature_dim,
                dtype=torch.float64,
            )

        x64 = x.double()

        sum_x += x64.sum(dim=0)
        sum_x2 += (x64 * x64).sum(dim=0)
        count += x64.size(0)

    if count == 0 or sum_x is None or sum_x2 is None:
        return None, None

    mean = sum_x / count

    variance = (
        sum_x2 / count
        - mean * mean
    )

    variance = torch.clamp(
        variance,
        min=1e-12,
    )

    std = torch.sqrt(variance)

    return (
        mean.float().unsqueeze(0),
        std.float().unsqueeze(0),
    )


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------

def load_upfd_data(
    root: str = "./data/UPFD",
    name: str = "politifact",
    feature: str = "bert",
    multimodal: bool = True,
    batch_size: int = 16,
    use_synthetic_fallback: bool = False,
    require_timestamps: bool = False,
) -> Tuple[
    Dataset,
    Dataset,
    Dataset,
    Dict[str, Union[int, float, str, bool, None]],
]:
    """
    Load train/validation/test UPFD datasets.

    Synthetic fallback is disabled by default and should never be used for
    reported research results.
    """

    try:
        train_ds = MultimodalUPFDDataset(
            root=root,
            name=name,
            split="train",
            feature=feature,
            multimodal=multimodal,
        )

        val_ds = MultimodalUPFDDataset(
            root=root,
            name=name,
            split="val",
            feature=feature,
            multimodal=multimodal,
        )

        test_ds = MultimodalUPFDDataset(
            root=root,
            name=name,
            split="test",
            feature=feature,
            multimodal=multimodal,
        )

        # -------------------------------------------------------------
        # Compute profile normalization from TRAINING data only.
        # -------------------------------------------------------------
        if multimodal and train_ds.profile_ds is not None:
            profile_mean, profile_std = (
                _compute_training_profile_stats(train_ds)
            )

            train_ds.set_profile_stats(
                profile_mean,
                profile_std,
            )
            val_ds.set_profile_stats(
                profile_mean,
                profile_std,
            )
            test_ds.set_profile_stats(
                profile_mean,
                profile_std,
            )

        # Verify first graph.
        sample = train_ds[0]

        if sample.x is None:
            raise ValueError(
                "UPFD sample has no node feature matrix."
            )

        in_dim = int(sample.x.size(-1))
        num_classes = 2

        timestamp_mapping_available = (
            train_ds.timestamps_available()
            and val_ds.timestamps_available()
            and test_ds.timestamps_available()
        )

        if require_timestamps and not timestamp_mapping_available:
            raise RuntimeError(
                "Real UPFD timestamps are required, but the official "
                "timestamp mapping could not be loaded for all splits."
            )

    except Exception as exc:
        if not use_synthetic_fallback:
            raise RuntimeError(
                "\nUPFD dataset initialization failed.\n"
                f"Dataset: {name}\n"
                f"Root: {root}\n"
                f"Reason: {exc}\n\n"
                "Synthetic fallback is disabled for research safety. "
                "Fix the UPFD installation/data files before continuing."
            ) from exc

        print(
            "WARNING: Using SYNTHETIC data for debugging only.\n"
            f"Original UPFD loading error: {exc}"
        )

        train_ds = SyntheticCascadeDataset(
            num_graphs=60,
            feature_dim=768,
            seed=42,
        )

        val_ds = SyntheticCascadeDataset(
            num_graphs=30,
            feature_dim=768,
            seed=43,
        )

        test_ds = SyntheticCascadeDataset(
            num_graphs=120,
            feature_dim=768,
            seed=44,
        )

        sample = train_ds[0]

        in_dim = int(sample.x.size(-1))
        num_classes = 2

        timestamp_mapping_available = False

    metadata = {
        "dataset_name": name,
        "feature_type": feature,
        "multimodal": multimodal,
        "input_dim": in_dim,
        "num_classes": num_classes,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "timestamp_mapping_available": timestamp_mapping_available,
        "research_data": not use_synthetic_fallback,
    }

    return (
        train_ds,
        val_ds,
        test_ds,
        metadata,
    )


# ---------------------------------------------------------------------
# PyG DataLoaders
# ---------------------------------------------------------------------

def get_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 16,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch Geometric DataLoaders."""

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )