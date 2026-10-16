"""
Rumor Centrality and Jordan Center algorithms for cascade source node identification.
Implements Shah & Zaman (2011, 2012) linear-time message-passing maximum likelihood
source estimation on propagation trees.
"""
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import torch


def compute_subtree_sizes(T: nx.DiGraph, root: int) -> Dict[int, int]:
    """Computes subtree sizes for all nodes when T is viewed with 'root' as origin."""
    undir = T.to_undirected()
    if not nx.is_connected(undir):
        cc = max(nx.connected_components(undir), key=len)
        undir = undir.subgraph(cc).copy()
    root = root if root in undir else next(iter(undir.nodes()))
    tree = nx.bfs_tree(undir, root)
    top_order = list(nx.topological_sort(tree))
    subtree_sizes = {node: 1 for node in undir.nodes()}
    for u in top_order[::-1]:
        for child in tree.successors(u):
            subtree_sizes[u] += subtree_sizes[child]
    return subtree_sizes


def compute_rumor_centrality_all_nodes(G: nx.DiGraph, root_hint: int = 0) -> Dict[int, float]:
    """
    Computes log rumor centrality for all nodes in tree G in O(N) time using Shah-Zaman message passing.
    log R(w) = log R(u) + log(T_w) - log(N - T_w) for adjacent nodes (u, w).
    """
    n = G.number_of_nodes()
    if n <= 1:
        return {node: 1.0 for node in G.nodes()}

    G_undir = G.to_undirected()
    if not nx.is_connected(G_undir):
        # Handle largest connected component
        cc = max(nx.connected_components(G_undir), key=len)
        G_undir = G_undir.subgraph(cc).copy()
        n = G_undir.number_of_nodes()

    # Pick root node
    v0 = root_hint if root_hint in G_undir else next(iter(G_undir.nodes()))

    # Build directed BFS tree from v0
    tree = nx.bfs_tree(G_undir, v0)
    top_order = list(nx.topological_sort(tree))
    post_order = top_order[::-1]

    # Compute subtree sizes rooted at v0
    subtree_sizes = {node: 1 for node in G_undir.nodes()}
    for u in post_order:
        for child in tree.successors(u):
            subtree_sizes[u] += subtree_sizes[child]

    # Base log rumor score at v0: -sum_{u} log(T_u)
    log_scores = {}
    log_scores[v0] = -sum(np.log(max(1, subtree_sizes[u])) for u in G_undir.nodes())

    # Forward message passing from v0 down the tree: O(N)
    for u in top_order:
        for w in tree.successors(u):
            t_w = subtree_sizes[w]
            # When rooted at w, u's branch has size (n - t_w)
            log_scores[w] = log_scores[u] + np.log(max(1, t_w)) - np.log(max(1, n - t_w))

    # Normalize to [0, 1] range
    scores = {node: float(log_scores.get(node, -1e6)) for node in G.nodes()}
    max_val = max(scores.values())
    min_val = min(scores.values())
    if max_val > min_val:
        norm_scores = {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
    else:
        norm_scores = {k: 1.0 / len(scores) for k in scores}

    return norm_scores


def compute_jordan_center(G: nx.DiGraph) -> Tuple[int, Dict[int, int]]:
    """
    Computes Jordan Center: node with minimum eccentricity (min max-distance to any node).
    Returns (jordan_center_node, eccentricities_dict).
    """
    G_undir = G.to_undirected()
    if G_undir.number_of_nodes() == 0:
        return 0, {}

    if not nx.is_connected(G_undir):
        largest_cc = max(nx.connected_components(G_undir), key=len)
        G_undir = G_undir.subgraph(largest_cc)

    eccentricities = nx.eccentricity(G_undir)
    jordan_node = min(eccentricities, key=eccentricities.get)
    return jordan_node, eccentricities


def find_top_k_rumor_sources(G: nx.DiGraph, k: int = 5) -> List[Tuple[int, float]]:
    """Returns top-k candidate source nodes ranked by rumor centrality."""
    scores = compute_rumor_centrality_all_nodes(G)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]
