"""
Unit tests for Explainable AI (XAI) evaluation module.

Tests cover:
1. Input snapshots and tensors are never mutated.
2. Original predictions and model evaluation state remain unchanged after explanation calculations.
3. Modality masking addresses the correct feature dimensions (both standard 782-dim and custom).
4. Node masking respects active snapshots and preserves graph structure (edge_index, active_mask).
5. Root node 0 is handled separately from propagating user nodes.
6. Confidence-difference signs follow the documented convention: delta_p = p_base - p_pert.
7. Invalid inputs, dimension mismatches, and empty/single-node cascades are handled safely.
8. Repeated runs with identical inputs are deterministic.
9. Faithfulness evaluation handles small graphs and dynamically caps top-k values.
10. Comprehensive cascade explanation wrapper correctly integrates and ranks attributions.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
from torch_geometric.data import Data

from src.models.stgnn import SpatioTemporalGNN
from src.evaluation.xai import (
    DEFAULT_FEATURE_GROUPS,
    compute_modality_attribution,
    compute_node_attribution,
    evaluate_explanation_faithfulness,
    compute_cascade_explanations,
)


class TestXAIEvaluation(unittest.TestCase):
    """Test suite for post-hoc STGNN XAI attribution and faithfulness evaluation."""

    def setUp(self):
        """Set up small deterministic models and synthetic snapshot cascades."""
        torch.manual_seed(42)
        np.random.seed(42)

        # 1. Small STGNN for fast tests (custom in_features=16)
        self.small_in_dim = 16
        self.hidden_dim = 16
        self.num_snapshots = 3

        self.small_model = SpatioTemporalGNN(
            in_features=self.small_in_dim,
            hidden_dim=self.hidden_dim,
            num_heads=2,
            num_snapshots=self.num_snapshots,
            temporal_cell="gru",
            num_classes=2,
            dropout=0.0,
        )
        self.small_model.eval()

        # Custom feature groups matching small_in_dim
        self.custom_groups = {
            "group_a": (0, 8),
            "group_b": (8, 12),
            "group_c": (12, 16),
        }

        # 2. Synthetic cascade with 4 nodes across 3 snapshots
        # Node 0 = root (active all 3 snapshots)
        # Node 1 = user (active in snap 1, 2)
        # Node 2 = user (active in snap 2)
        # Node 3 = user (inactive all snapshots)
        self.num_nodes = 4
        self.snaps = self._build_synthetic_snapshots(
            num_nodes=self.num_nodes,
            in_dim=self.small_in_dim,
            num_snapshots=self.num_snapshots,
        )

        # 3. Model and snapshots for standard 782-dim layout
        self.std_in_dim = 782
        self.std_model = SpatioTemporalGNN(
            in_features=self.std_in_dim,
            hidden_dim=16,
            num_heads=2,
            num_snapshots=2,
            temporal_cell="gru",
            num_classes=2,
            dropout=0.0,
        )
        self.std_model.eval()
        self.std_snaps = self._build_synthetic_snapshots(
            num_nodes=3,
            in_dim=self.std_in_dim,
            num_snapshots=2,
        )

    def _build_synthetic_snapshots(
        self,
        num_nodes: int,
        in_dim: int,
        num_snapshots: int,
    ) -> list[Data]:
        """Construct synthetic chronological snapshots with tree edges and active masks."""
        snaps = []
        # Tree edges: 0 -> 1, 1 -> 2, 0 -> 3 (if present)
        edges_list = []
        if num_nodes > 1:
            edges_list.append([0, 1])
        if num_nodes > 2:
            edges_list.append([1, 2])
        if num_nodes > 3:
            edges_list.append([0, 3])

        all_edges = (
            torch.tensor(edges_list, dtype=torch.long).t()
            if edges_list
            else torch.empty((2, 0), dtype=torch.long)
        )

        for t in range(num_snapshots):
            x = torch.randn(num_nodes, in_dim, dtype=torch.float32)
            active_mask = torch.zeros(num_nodes, dtype=torch.bool)
            active_mask[0] = True  # root always active

            if num_nodes > 1 and t >= 1:
                active_mask[1] = True
            if num_nodes > 2 and t >= 2:
                active_mask[2] = True

            # Inactive nodes have zero features
            x[~active_mask] = 0.0

            # Filter edges to active nodes
            if all_edges.numel() > 0:
                edge_mask = active_mask[all_edges[0]] & active_mask[all_edges[1]]
                snap_edges = all_edges[:, edge_mask]
            else:
                snap_edges = torch.empty((2, 0), dtype=torch.long)

            snaps.append(
                Data(
                    x=x,
                    edge_index=snap_edges,
                    active_mask=active_mask,
                    node_timestamps=torch.tensor([float("nan")] + [float(i * 100) for i in range(1, num_nodes)]),
                )
            )
        return snaps

    # -----------------------------------------------------------------
    # Test 1: Non-mutation of inputs
    # -----------------------------------------------------------------

    def test_input_snapshots_never_mutated(self):
        """Verifies that running XAI explanations leaves input tensors bit-for-bit unchanged."""
        # Deepcopy snapshots tensors before running
        original_x = [s.x.clone() for s in self.snaps]
        original_edges = [s.edge_index.clone() for s in self.snaps]
        original_masks = [s.active_mask.clone() for s in self.snaps]

        # Run all three XAI routines
        _ = compute_modality_attribution(
            self.small_model, self.snaps, feature_groups=self.custom_groups
        )
        _ = compute_node_attribution(self.small_model, self.snaps)
        _ = evaluate_explanation_faithfulness(
            self.small_model, self.snaps, top_k_list=(1, 2)
        )

        # Assert every tensor is identical
        for t in range(len(self.snaps)):
            self.assertTrue(
                torch.equal(self.snaps[t].x, original_x[t]),
                f"Snapshot {t} x was mutated!",
            )
            self.assertTrue(
                torch.equal(self.snaps[t].edge_index, original_edges[t]),
                f"Snapshot {t} edge_index was mutated!",
            )
            self.assertTrue(
                torch.equal(self.snaps[t].active_mask, original_masks[t]),
                f"Snapshot {t} active_mask was mutated!",
            )

    # -----------------------------------------------------------------
    # Test 2: Original predictions and model state unchanged
    # -----------------------------------------------------------------

    def test_original_predictions_unchanged(self):
        """Verifies that model logits before and after XAI evaluation are identical."""
        with torch.no_grad():
            out_before = self.small_model.forward_snapshots(self.snaps)[0]

        _ = compute_cascade_explanations(
            self.small_model,
            self.snaps,
            feature_groups=self.custom_groups,
            evaluate_faithfulness=True,
            faithfulness_k_list=(1, 2),
        )

        with torch.no_grad():
            out_after = self.small_model.forward_snapshots(self.snaps)[0]

        self.assertTrue(
            torch.allclose(out_before, out_after, atol=1e-7),
            "Model output changed after computing explanations!",
        )
        self.assertFalse(self.small_model.training, "Model was left in training mode!")

    # -----------------------------------------------------------------
    # Test 3: Modality masking addresses correct feature dimensions
    # -----------------------------------------------------------------

    def test_modality_masking_dimensions(self):
        """Verifies modality attribution on standard 782-dim and custom configurations."""
        # Standard layout: 0..768 (BERT), 768..778 (Profile), 778..782 (Temporal)
        res_std = compute_modality_attribution(self.std_model, self.std_snaps)
        self.assertIn("bert_content", res_std["modality_attributions"])
        self.assertIn("user_profile", res_std["modality_attributions"])
        self.assertIn("temporal_dynamics", res_std["modality_attributions"])

        bert_info = res_std["modality_attributions"]["bert_content"]
        self.assertEqual(bert_info["feature_slice"], [0, 768])
        prof_info = res_std["modality_attributions"]["user_profile"]
        self.assertEqual(prof_info["feature_slice"], [768, 778])
        temp_info = res_std["modality_attributions"]["temporal_dynamics"]
        self.assertEqual(temp_info["feature_slice"], [778, 782])

        # Custom layout
        res_custom = compute_modality_attribution(
            self.small_model, self.snaps, feature_groups=self.custom_groups
        )
        self.assertIn("group_a", res_custom["modality_attributions"])
        self.assertEqual(res_custom["modality_attributions"]["group_a"]["feature_slice"], [0, 8])

    # -----------------------------------------------------------------
    # Test 4: Node masking respects active snapshots and preserves graph
    # -----------------------------------------------------------------

    def test_node_masking_active_snapshots_and_graph(self):
        """Verifies that node attribution tracks active snapshot counts and preserves topology."""
        res = compute_node_attribution(self.small_model, self.snaps)

        # Node 1 is active in snap 1 and 2 (count = 2)
        # Node 2 is active only in snap 2 (count = 1)
        # Node 3 is active in 0 snapshots (not in user_attributions)
        node_attrs = res["node_attributions"]
        self.assertIn(1, node_attrs)
        self.assertEqual(node_attrs[1]["active_snapshots_count"], 2)

        self.assertIn(2, node_attrs)
        self.assertEqual(node_attrs[2]["active_snapshots_count"], 1)

        self.assertNotIn(3, node_attrs)

        # Root node 0 active in all 3 snapshots
        root_attr = res["root_attribution"]
        self.assertIsNotNone(root_attr)
        self.assertEqual(root_attr["active_snapshots_count"], 3)

    # -----------------------------------------------------------------
    # Test 5: Root node 0 handled separately from user nodes
    # -----------------------------------------------------------------

    def test_root_node_separation(self):
        """Verifies root node 0 is excluded from node_attributions by default."""
        res_default = compute_node_attribution(
            self.small_model, self.snaps, include_root=False
        )
        self.assertNotIn(0, res_default["node_attributions"])
        self.assertIsNotNone(res_default["root_attribution"])
        self.assertEqual(res_default["root_attribution"]["node_id"], 0)
        self.assertIn("Root News Item", res_default["root_attribution"]["role"])

        # When include_root=True, node 0 also appears in node_attributions
        res_with_root = compute_node_attribution(
            self.small_model, self.snaps, include_root=True
        )
        self.assertIn(0, res_with_root["node_attributions"])

    # -----------------------------------------------------------------
    # Test 6: Confidence-difference sign convention
    # -----------------------------------------------------------------

    def test_confidence_difference_sign_convention(self):
        """Verifies delta_p = base_prob - pert_prob matches numerically."""
        # Modality attribution
        res_mod = compute_modality_attribution(
            self.small_model, self.snaps, feature_groups=self.custom_groups
        )
        base_p = res_mod["baseline_probability"]
        for g_name, g_data in res_mod["modality_attributions"].items():
            expected_delta = base_p - g_data["perturbed_probability"]
            self.assertAlmostEqual(g_data["delta_probability"], expected_delta, places=6)

        # Node attribution
        res_node = compute_node_attribution(self.small_model, self.snaps)
        for nid, n_data in res_node["node_attributions"].items():
            expected_delta = base_p - n_data["perturbed_probability"]
            self.assertAlmostEqual(n_data["delta_probability"], expected_delta, places=6)

    # -----------------------------------------------------------------
    # Test 7: Invalid inputs and edge cases
    # -----------------------------------------------------------------

    def test_invalid_inputs_and_edge_cases(self):
        """Verifies descriptive ValueErrors on empty, invalid, or mismatched inputs."""
        # Empty snapshots
        with self.assertRaises(ValueError):
            compute_modality_attribution(self.small_model, [])

        with self.assertRaises(ValueError):
            compute_node_attribution(self.small_model, [])

        # Invalid target class
        with self.assertRaises(ValueError):
            compute_modality_attribution(self.small_model, self.snaps, target_class=2)

        with self.assertRaises(ValueError):
            compute_node_attribution(self.small_model, self.snaps, target_class=-1)

        # Dimension mismatch for default groups (small_in_dim=16 < 782)
        with self.assertRaises(ValueError):
            compute_modality_attribution(self.small_model, self.snaps)

        # Invalid custom group slices
        with self.assertRaises(ValueError):
            compute_modality_attribution(
                self.small_model, self.snaps, feature_groups={"bad": (10, 25)}
            )

        # Single-node cascade (only root node 0)
        single_node_snaps = self._build_synthetic_snapshots(
            num_nodes=1, in_dim=self.small_in_dim, num_snapshots=self.num_snapshots
        )
        res_single = compute_node_attribution(self.small_model, single_node_snaps)
        self.assertEqual(len(res_single["node_attributions"]), 0)
        self.assertIsNotNone(res_single["root_attribution"])
        self.assertEqual(res_single["total_active_user_nodes"], 0)

    # -----------------------------------------------------------------
    # Test 8: Determinism of repeated runs
    # -----------------------------------------------------------------

    def test_repeated_runs_deterministic(self):
        """Verifies that consecutive runs with identical inputs produce identical values."""
        res1 = compute_modality_attribution(
            self.small_model, self.snaps, feature_groups=self.custom_groups
        )
        res2 = compute_modality_attribution(
            self.small_model, self.snaps, feature_groups=self.custom_groups
        )
        self.assertAlmostEqual(res1["baseline_probability"], res2["baseline_probability"], places=7)
        for g in self.custom_groups:
            self.assertAlmostEqual(
                res1["modality_attributions"][g]["delta_probability"],
                res2["modality_attributions"][g]["delta_probability"],
                places=7,
            )

        node_res1 = compute_node_attribution(self.small_model, self.snaps)
        node_res2 = compute_node_attribution(self.small_model, self.snaps)
        for nid in node_res1["node_attributions"]:
            self.assertAlmostEqual(
                node_res1["node_attributions"][nid]["delta_probability"],
                node_res2["node_attributions"][nid]["delta_probability"],
                places=7,
            )

    # -----------------------------------------------------------------
    # Test 9: Faithfulness evaluation handles small graphs and valid k
    # -----------------------------------------------------------------

    def test_faithfulness_evaluation_small_graphs_and_top_k(self):
        """Verifies that faithfulness evaluation caps k values and produces curves."""
        # Only 2 user nodes are active (nodes 1 and 2)
        res_faith = evaluate_explanation_faithfulness(
            self.small_model,
            self.snaps,
            top_k_list=(1, 2, 5, 10),
            seed=42,
        )

        # k values should be capped by eligible user nodes (2)
        self.assertEqual(res_faith["total_eligible_nodes"], 2)
        self.assertEqual(res_faith["evaluated_k_values"], [1, 2])

        # Verify deletion curves exist and have correct keys
        for curve_key in ("attribution_deletion_curve", "random_deletion_curve", "degree_deletion_curve"):
            self.assertIn(curve_key, res_faith)
            curve = res_faith[curve_key]
            self.assertEqual(len(curve), 2)
            self.assertEqual(curve[0]["k"], 1)
            self.assertEqual(curve[1]["k"], 2)
            self.assertIn("confidence_drop", curve[0])
            self.assertIn("perturbed_probability", curve[0])

        # Single-node cascade handling in faithfulness
        single_node_snaps = self._build_synthetic_snapshots(
            num_nodes=1, in_dim=self.small_in_dim, num_snapshots=self.num_snapshots
        )
        res_faith_single = evaluate_explanation_faithfulness(
            self.small_model, single_node_snaps, top_k_list=(1, 2)
        )
        self.assertEqual(res_faith_single["total_eligible_nodes"], 0)
        self.assertEqual(res_faith_single["evaluated_k_values"], [])

    # -----------------------------------------------------------------
    # Test 10: Comprehensive cascade explanation wrapper
    # -----------------------------------------------------------------

    def test_cascade_explanation_wrapper(self):
        """Verifies compute_cascade_explanations packages attributions and rankings cleanly."""
        res = compute_cascade_explanations(
            self.small_model,
            self.snaps,
            feature_groups=self.custom_groups,
            evaluate_faithfulness=True,
            faithfulness_k_list=(1, 2),
        )

        self.assertIn("target_class", res)
        self.assertIn("baseline_probability", res)
        self.assertIn("modality_attributions", res)
        self.assertIn("root_attribution", res)
        self.assertIn("top_supporting_user_nodes", res)
        self.assertIn("top_countervailing_user_nodes", res)
        self.assertIn("faithfulness", res)
        self.assertIn("methodological_notes", res)

        # Also test on standard 782-dim model with default feature groups
        res_std = compute_cascade_explanations(
            self.std_model,
            self.std_snaps,
            evaluate_faithfulness=True,
            faithfulness_k_list=(1, 2),
        )
        self.assertIn("bert_content", res_std["modality_attributions"])
        self.assertIn("user_profile", res_std["modality_attributions"])
        self.assertIn("temporal_dynamics", res_std["modality_attributions"])


if __name__ == "__main__":
    unittest.main()
