"""
Structural and geometric cascade feature extractors for propagation trees.
Computes structural virality (Wiener index), depth, breadth, and diffusion dynamics.
"""
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import torch


def build_networkx_tree(edge_index: torch.Tensor, num_nodes: int = None) -> nx.DiGraph:
    """Converts PyG edge_index to a NetworkX directed graph."""
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
    return G


def compute_structural_virality(G: nx.Graph) -> float:
    """
    Computes Wiener index / structural virality v(T):
    v(T) = (1 / (n * (n - 1))) * sum_{i != j} d(i, j)
    where d(i, j) is the shortest path length between nodes i and j in the undirected graph.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    G_undir = G.to_undirected()
    if not nx.is_connected(G_undir):
        # Handle disconnected components by averaging over connected components
        total_dist = 0.0
        pairs = 0
        for comp in nx.connected_components(G_undir):
            sub = G_undir.subgraph(comp)
            sub_n = len(comp)
            if sub_n > 1:
                path_lengths = dict(nx.all_pairs_shortest_path_length(sub))
                for src in path_lengths:
                    for dst in path_lengths[src]:
                        total_dist += path_lengths[src][dst]
                pairs += sub_n * (sub_n - 1)
        return float(total_dist / pairs) if pairs > 0 else 0.0

    path_lengths = dict(nx.all_pairs_shortest_path_length(G_undir))
    total_dist = sum(sum(lengths.values()) for lengths in path_lengths.values())
    return float(total_dist / (n * (n - 1)))


def compute_tree_levels(G: nx.DiGraph, root_node: int = 0) -> Dict[int, int]:
    """Computes depth level of each node starting from root_node."""
    levels = {node: 0 for node in G.nodes()}
    try:
        lengths = nx.single_source_shortest_path_length(G, root_node)
        levels.update(lengths)
    except Exception:
        # Fallback using undirected search
        G_undir = G.to_undirected()
        lengths = nx.single_source_shortest_path_length(G_undir, root_node)
        levels.update(lengths)
    return levels


def compute_cascade_metrics(edge_index: torch.Tensor, num_nodes: int = None, root_node: int = 0) -> Dict[str, float]:
    """
    Extracts high-level topological cascade metrics:
    - cascade_size: Number of nodes
    - max_depth: Deepest level reached
    - max_breadth: Maximum nodes in any generation
    - structural_virality: Average all-pairs shortest path distance
    - out_degree_max: Highest fan-out from an individual user
    - out_degree_mean: Average fan-out
    - leaf_ratio: Ratio of terminal nodes (leaves)
    """
    G = build_networkx_tree(edge_index, num_nodes)
    n = G.number_of_nodes()
    if n == 0:
        return {
            "cascade_size": 0.0,
            "max_depth": 0.0,
            "max_breadth": 0.0,
            "structural_virality": 0.0,
            "out_degree_max": 0.0,
            "out_degree_mean": 0.0,
            "leaf_ratio": 0.0,
        }

    levels = compute_tree_levels(G, root_node)
    max_depth = max(levels.values()) if levels else 0

    # Breadth per level
    level_counts = {}
    for node, lvl in levels.items():
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    max_breadth = max(level_counts.values()) if level_counts else 0

    # Degree stats
    out_degrees = [d for _, d in G.out_degree()]
    out_max = max(out_degrees) if out_degrees else 0
    out_mean = float(np.mean(out_degrees)) if out_degrees else 0.0

    # Leaves (out-degree 0)
    leaves = [node for node, d in G.out_degree() if d == 0]
    leaf_ratio = len(leaves) / n if n > 0 else 0.0

    virality = compute_structural_virality(G)

    return {
        "cascade_size": float(n),
        "max_depth": float(max_depth),
        "max_breadth": float(max_breadth),
        "structural_virality": virality,
        "out_degree_max": float(out_max),
        "out_degree_mean": out_mean,
        "leaf_ratio": leaf_ratio,
    }
