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
        edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]], dtype=torch.long)
        x = torch.randn(5, 32)
        y = torch.tensor([1], dtype=torch.long)
        self.sample_data = Data(x=x, edge_index=edge_index, y=y)

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
        # 2 cascades in a batch
        g1 = Data(x=torch.randn(6, 64), edge_index=torch.tensor([[0, 0, 1, 2, 2], [1, 2, 3, 4, 5]]), y=torch.tensor([1]))
        g2 = Data(x=torch.randn(4, 64), edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]]), y=torch.tensor([0]))
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
        self.data = Data(x=torch.randn(5, 16), edge_index=edge_index, y=torch.tensor([1]))

    def test_rumor_centrality(self):
        G = build_networkx_tree(self.data.edge_index, 5)
        scores = compute_rumor_centrality_all_nodes(G)
        self.assertEqual(len(scores), 5)
        # Root node 0 has the highest rumor centrality score
        top_node = max(scores, key=scores.get)
        self.assertEqual(top_node, 0)

    def test_source_identifier_evaluation(self):
        identifier = CascadeSourceIdentifier(hybrid_weight=0.5)
        res = identifier.evaluate_cascade_source(self.data, true_source=0, method="rumor_centrality")
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


if __name__ == "__main__":
    unittest.main()
