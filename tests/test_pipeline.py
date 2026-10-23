"""
Unit and integration test suite for the Spatio-Temporal GNN Misinformation Detection system.
Tests data loaders, temporal snapshot generation, baseline and STGNN models, and source localization.
"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import networkx as nx

from src.data.cascade_features import (
    build_networkx_tree,
    compute_structural_virality,
    compute_tree_levels,
    compute_cascade_metrics,
)
from src.data.temporal_builder import TemporalCascadeBuilder
from src.data.dataset_loader import SyntheticCascadeDataset
from src.models.nlp_baseline import NLPContentClassifier
from src.models.static_gnn import StaticCascadeGNN
from src.models.stgnn import SpatioTemporalGNN
from src.algorithms.rumor_centrality import (
    compute_rumor_centrality_all_nodes,
    compute_jordan_center,
    find_top_k_rumor_sources,
)
from src.algorithms.bot_detection import BotNetworkIsolator
from src.models.source_identifier import CascadeSourceIdentifier
from src.evaluation.metrics import compute_classification_metrics, aggregate_source_localization_metrics


class TestDataAndTemporalPipeline(unittest.TestCase):
    def setUp(self):
        # Create a small tree graph: 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
        # Node 0 is the news claim (timestamp NaN); nodes 1..4 have valid chronological timestamps
        edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]], dtype=torch.long)
        x = torch.randn(5, 32)
        y = torch.tensor([1], dtype=torch.long)
        node_timestamps = torch.tensor([float("nan"), 100.0, 200.0, 300.0, 400.0], dtype=torch.float64)
        self.sample_data = Data(x=x, edge_index=edge_index, y=y, node_timestamps=node_timestamps)

    def test_cascade_features(self):
        metrics = compute_cascade_metrics(self.sample_data.edge_index, num_nodes=5)
        self.assertEqual(metrics["cascade_size"], 5.0)
        self.assertEqual(metrics["max_depth"], 2.0)
        self.assertGreater(metrics["structural_virality"], 0.0)
        self.assertEqual(metrics["out_degree_max"], 2.0)

    def test_temporal_builder(self):
        builder = TemporalCascadeBuilder(num_snapshots=3, add_temporal_features=True)
        snapshots = builder.build_snapshots(self.sample_data)
        self.assertEqual(len(snapshots), 3)

        # In last snapshot, all nodes must be active
        last_snap = snapshots[-1]
        self.assertTrue(last_snap.active_mask.all())
        # Check augmented features: 32 + 4 = 36
        self.assertEqual(last_snap.x.size(-1), 36)

    def test_synthetic_dataset(self):
        ds = SyntheticCascadeDataset(num_graphs=10, feature_dim=64)
        self.assertEqual(len(ds), 10)
        sample = ds[0]
        self.assertEqual(sample.x.size(-1), 64)
        self.assertIn(sample.y.item(), [0, 1])


class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        # 2 cascades in a batch with valid UPFD timestamp metadata
        g1 = Data(
            x=torch.randn(6, 64),
            edge_index=torch.tensor([[0, 0, 1, 2, 2], [1, 2, 3, 4, 5]]),
            y=torch.tensor([1]),
            node_timestamps=torch.tensor([float("nan"), 100.0, 200.0, 300.0, 400.0, 500.0], dtype=torch.float64),
        )
        g2 = Data(
            x=torch.randn(4, 64),
            edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]),
            y=torch.tensor([0]),
            node_timestamps=torch.tensor([float("nan"), 100.0, 200.0, 300.0], dtype=torch.float64),
        )
        self.batch = Batch.from_data_list([g1, g2])
        self.single_graph = g1

    def test_nlp_baseline(self):
        model = NLPContentClassifier(in_features=64, hidden_dim=32, num_classes=2)
        logits = model(self.batch)
        self.assertEqual(logits.shape, (2, 2))
        loss = nn.CrossEntropyLoss()(logits, self.batch.y)
        loss.backward()
        self.assertIsNotNone(model.classifier[0].weight.grad)

    def test_static_gnn(self):
        model = StaticCascadeGNN(in_features=64, hidden_dim=32, gnn_type="gat", num_classes=2)
        logits = model(self.batch)
        self.assertEqual(logits.shape, (2, 2))
        loss = nn.CrossEntropyLoss()(logits, self.batch.y)
        loss.backward()
        self.assertIsNotNone(model.input_proj.weight.grad)

    def test_stgnn_forward_and_backward(self):
        builder = TemporalCascadeBuilder(num_snapshots=3, add_temporal_features=True)
        snapshots = builder.build_snapshots(self.single_graph)

        in_dim = snapshots[0].x.size(-1)
        stgnn = SpatioTemporalGNN(in_features=in_dim, hidden_dim=32, num_snapshots=3, num_classes=2)

        veracity_logits, source_scores, traj = stgnn.forward_snapshots(snapshots)
        self.assertEqual(veracity_logits.shape, (1, 2))
        self.assertEqual(source_scores.shape, (self.single_graph.x.size(0), 1))
        self.assertEqual(len(traj), 3)

        # Test joint backward pass
        loss = veracity_logits.sum() + source_scores.sum()
        loss.backward()
        self.assertIsNotNone(stgnn.input_proj.weight.grad)


class TestAlgorithmsAndSourceIdentification(unittest.TestCase):
    def setUp(self):
        # Tree: 0 is center/root: 0->1, 0->2, 0->3, 1->4
        edge_index = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 4]], dtype=torch.long)
        node_timestamps = torch.tensor([float("nan"), 100.0, 200.0, 300.0, 400.0], dtype=torch.float64)
        self.data = Data(x=torch.randn(5, 16), edge_index=edge_index, y=torch.tensor([1]), node_timestamps=node_timestamps)

    def test_rumor_centrality(self):
        G = build_networkx_tree(self.data.edge_index, 5)
        scores = compute_rumor_centrality_all_nodes(G)
        self.assertEqual(len(scores), 5)
        # Root node 0 has the highest rumor centrality score
        top_node = max(scores, key=scores.get)
        self.assertEqual(top_node, 0)

    def test_source_identifier_evaluation(self):
        identifier = CascadeSourceIdentifier(hybrid_weight=0.5)
        # In UPFD, node 0 is the news claim and cannot be a user source target.
        # User node 1 has the highest rumor centrality among candidate users (nodes 1..4).
        res = identifier.evaluate_cascade_source(self.data, true_source=1, method="rumor_centrality")
        self.assertEqual(res["true_rank"], 1)
        self.assertEqual(res["hit@1"], 1.0)
        self.assertEqual(res["error_distance"], 0.0)

    def test_bot_network_isolation(self):
        # Create hub node with high fanout
        edge_index = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]], dtype=torch.long)
        hub_data = Data(x=torch.randn(7, 16), edge_index=edge_index, y=torch.tensor([1]))
        G = build_networkx_tree(hub_data.edge_index, 7)
        isolator = BotNetworkIsolator(min_fanout=3)
        bot_res = isolator.isolate_bot_rings(hub_data, G)
        # Node 1 should be isolated as a bot hub
        self.assertIn(1, bot_res["suspicious_nodes"])


class TestTemporalLeakageRegression(unittest.TestCase):
    """
    Deterministic regression tests verifying the resolution of temporal leakage.
    Ensures that early-stage snapshots depend strictly on prefix-observed data
    and observation fractions, not future cascade duration or endpoint timestamps.
    """

    def setUp(self):
        # 1 root (node 0) + 8 users (nodes 1..8)
        self.num_nodes = 9
        # Tree: 0->1, 1->2, 1->3, 2->4, 2->5, 3->6, 3->7, 7->8
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 7],
             [1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=torch.long,
        )
        self.base_x = torch.randn(self.num_nodes, 778)
        self.timestamps = torch.tensor(
            [float("nan"), 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            dtype=torch.float64,
        )
        self.cascade = Data(
            x=self.base_x.clone(),
            edge_index=self.edge_index.clone(),
            y=torch.tensor([1], dtype=torch.long),
            node_timestamps=self.timestamps.clone(),
        )

    def test_observation_fraction_values(self):
        """At each observation fraction, progression feature must exactly match the predefined fraction."""
        builder = TemporalCascadeBuilder(num_snapshots=4, add_temporal_features=True)
        snapshots = builder.build_snapshots(self.cascade)
        self.assertEqual(len(snapshots), 4)

        expected_fractions = [0.25, 0.50, 0.75, 1.00]
        for t, expected_f in enumerate(expected_fractions):
            snap = snapshots[t]
            self.assertAlmostEqual(snap.snapshot_fraction.item(), expected_f, places=6)
            self.assertAlmostEqual(snap.observation_fraction.item(), expected_f, places=6)
            self.assertAlmostEqual(snap.elapsed_fraction.item(), expected_f, places=6)

            active_mask = snap.active_mask
            active_indices = torch.where(active_mask)[0]

            # Root (0) must always be active
            self.assertTrue(active_mask[0].item())

            # For every active node, progression feature (dim 780, or -2) must equal expected_f
            for node_idx in active_indices:
                prog_val = snap.x[node_idx, 780].item()
                self.assertAlmostEqual(prog_val, expected_f, places=6)
                # Active indicator (dim 781, or -1) must be 1.0
                self.assertEqual(snap.x[node_idx, 781].item(), 1.0)

            # Inactive nodes must be strictly zero across all feature dimensions
            inactive_mask = ~active_mask
            if inactive_mask.any():
                self.assertTrue(torch.all(snap.x[inactive_mask] == 0.0).item())

    def test_invariance_to_future_duration_and_timestamps(self):
        """Prefix snapshot features must not depend on future timestamps or total cascade duration."""
        builder = TemporalCascadeBuilder(num_snapshots=4, add_temporal_features=True)

        # Cascade A: Normal duration (10s to 80s -> duration = 70s)
        cascade_a = self.cascade

        # Cascade B: Identical structure and initial nodes, but the final user's
        # arrival time is altered drastically (e.g. 10,000,000s -> duration = ~10Ms)
        ts_b = self.timestamps.clone()
        ts_b[8] = 10_000_000.0
        cascade_b = Data(
            x=self.base_x.clone(),
            edge_index=self.edge_index.clone(),
            y=torch.tensor([1], dtype=torch.long),
            node_timestamps=ts_b,
        )

        snaps_a = builder.build_snapshots(cascade_a)
        snaps_b = builder.build_snapshots(cascade_b)

        # Check prefix stages 0 (25%), 1 (50%), 2 (75%)
        for t in range(3):
            # Active masks must be identical
            self.assertTrue(torch.equal(snaps_a[t].active_mask, snaps_b[t].active_mask))
            # Observed edges must be identical
            self.assertTrue(torch.equal(snaps_a[t].edge_index, snaps_b[t].edge_index))
            # Node features must be completely invariant to future timestamp/duration
            self.assertTrue(
                torch.equal(snaps_a[t].x, snaps_b[t].x),
                f"Snapshot {t} leaked future duration into node features!",
            )

    def test_feature_dimensions_and_order(self):
        """Verify the 782-dim layout: 778 base + depth + out_degree + fraction + active."""
        builder = TemporalCascadeBuilder(num_snapshots=4, add_temporal_features=True)
        snapshots = builder.build_snapshots(self.cascade)

        for t, snap in enumerate(snapshots):
            self.assertEqual(snap.x.size(0), self.num_nodes)
            self.assertEqual(snap.x.size(1), 782)

            # For active nodes:
            active_nodes = torch.where(snap.active_mask)[0]
            for n in active_nodes:
                # Base features unchanged
                self.assertTrue(torch.equal(snap.x[n, :778], self.base_x[n]))
                # Depth in [0, 1]
                depth_norm = snap.x[n, 778].item()
                self.assertGreaterEqual(depth_norm, 0.0)
                self.assertLessEqual(depth_norm, 1.0)
                # Out-degree in [0, 1]
                deg_norm = snap.x[n, 779].item()
                self.assertGreaterEqual(deg_norm, 0.0)
                self.assertLessEqual(deg_norm, 1.0)
                # Progression fraction in (0, 1]
                self.assertAlmostEqual(snap.x[n, 780].item(), snap.snapshot_fraction.item(), places=6)
                # Active flag is 1.0
                self.assertEqual(snap.x[n, 781].item(), 1.0)

    def test_prefix_topology_and_per_snapshot_degree(self):
        """Out-degree must be calculated strictly from the observed subgraph, not the full cascade."""
        # Hub node 1 has 3 children: 2, 3, 4
        edge_index = torch.tensor([[0, 1, 1, 1], [1, 2, 3, 4]], dtype=torch.long)
        # Arrival: 1 arrives at 10, 2 at 20, 3 at 30, 4 at 40
        node_timestamps = torch.tensor([float("nan"), 10.0, 20.0, 30.0, 40.0], dtype=torch.float64)
        cascade = Data(
            x=torch.randn(5, 16),
            edge_index=edge_index,
            node_timestamps=node_timestamps,
        )

        builder = TemporalCascadeBuilder(num_snapshots=4, add_temporal_features=True)
        snapshots = builder.build_snapshots(cascade)

        # Monotonically nested masks
        for t in range(len(snapshots) - 1):
            subset = snapshots[t].active_mask <= snapshots[t + 1].active_mask
            self.assertTrue(subset.all().item())

        # In snapshot 0 (fraction 0.25 -> 1 user -> only node 0 and 1 active):
        # Node 1 has no active children yet, so its observed out-degree in snapshot 0 must be 0!
        snap_0 = snapshots[0]
        self.assertEqual(snap_0.edge_index.size(1), 1)  # Only edge 0 -> 1
        # Node 1 out-degree normalized feature (dim -3) must be 0.0
        self.assertEqual(snap_0.x[1, -3].item(), 0.0)

        # In snapshot 1 (fraction 0.50 -> 2 users -> nodes 0, 1, 2 active):
        # Edges observed: 0->1, 1->2. Node 1 has out-degree 1.
        snap_1 = snapshots[1]
        self.assertEqual(snap_1.edge_index.size(1), 2)

    def test_stgnn_model_compatibility(self):
        """Verify full-observation and prefix feature construction compatibility with SpatioTemporalGNN."""
        builder = TemporalCascadeBuilder(num_snapshots=4, add_temporal_features=True)
        snapshots = builder.build_snapshots(self.cascade)

        in_dim = snapshots[0].x.size(-1)
        self.assertEqual(in_dim, 782)

        model = SpatioTemporalGNN(in_features=in_dim, hidden_dim=32, num_snapshots=4, num_classes=2)
        model.eval()

        veracity_logits, source_scores, traj = model.forward_snapshots(snapshots)
        self.assertEqual(veracity_logits.shape, (1, 2))
        self.assertEqual(source_scores.shape, (self.num_nodes, 1))
        self.assertEqual(len(traj), 4)

        # Test gradient flow
        model.train()
        v_logits, s_scores, _ = model.forward_snapshots(snapshots)
        loss = v_logits.sum() + s_scores.sum()
        loss.backward()
        self.assertIsNotNone(model.input_proj.weight.grad)
        self.assertFalse(torch.isnan(model.input_proj.weight.grad).any())


if __name__ == "__main__":
    unittest.main()

