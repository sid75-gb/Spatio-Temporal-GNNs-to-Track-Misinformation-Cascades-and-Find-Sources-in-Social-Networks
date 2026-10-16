"""
Cascade visualization tools for misinformation propagation trees.
Generates publication-quality static tree diagrams and interactive web-ready graph representations.
"""
from typing import Dict, List, Optional, Tuple, Union
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from ..data.cascade_features import build_networkx_tree, compute_tree_levels


def hierarchical_layout(G: nx.DiGraph, root: int = 0) -> Dict[int, Tuple[float, float]]:
    """
    Computes a clean hierarchical tree layout with root at the top
    and subsequent generations placed along descending Y levels.
    """
    levels = compute_tree_levels(G, root_node=root)
    levels_dict: Dict[int, List[int]] = {}
    for node, lvl in levels.items():
        levels_dict.setdefault(lvl, []).append(node)

    pos = {}
    max_lvl = max(levels_dict.keys()) if levels_dict else 0
    for lvl, nodes in levels_dict.items():
        y = -(lvl)  # descending levels
        count = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (count - 1) / 2.0) * (1.0 / max(1, count * 0.2))
            pos[node] = (x, y)

    # For any node missing from pos (disconnected)
    for node in G.nodes():
        if node not in pos:
            pos[node] = (0.0, -max_lvl - 1)

    return pos


def plot_cascade_tree(
    edge_index: torch.Tensor,
    num_nodes: int,
    source_node: int = 0,
    bot_nodes: Optional[List[int]] = None,
    predicted_source: Optional[int] = None,
    title: str = "Misinformation Propagation Cascade",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a publication-ready tree visualization of the cascade.
    Color codes:
    - Gold / Star: True Origin Source Node
    - Crimson / Square: Coordinated Bot Amplifiers
    - DodgerBlue: Organic Retweeters
    - Purple Ring: Predicted Source
    """
    G = build_networkx_tree(edge_index, num_nodes)
    pos = hierarchical_layout(G, root=source_node)
    bot_set = set(bot_nodes) if bot_nodes else set()

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Draw edges with subtle opacity
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#475569",
        arrows=True,
        arrowsize=10,
        alpha=0.6,
        width=1.2,
    )

    # Partition nodes into categories
    regular_nodes = [n for n in G.nodes() if n != source_node and n not in bot_set]
    bots = [n for n in G.nodes() if n in bot_set and n != source_node]

    # Draw regular nodes
    if regular_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=regular_nodes, ax=ax,
            node_color="#38bdf8", node_size=120, alpha=0.85,
        )

    # Draw bot nodes
    if bots:
        nx.draw_networkx_nodes(
            G, pos, nodelist=bots, ax=ax,
            node_color="#ef4444", node_size=200, node_shape="s",
            label="Coordinated Bot Amplifier",
        )

    # Draw true source node
    if source_node in G:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[source_node], ax=ax,
            node_color="#fbbf24", node_size=350, node_shape="*",
            label=f"Origin Source (Node {source_node})",
        )

    # Draw predicted source indicator if different
    if predicted_source is not None and predicted_source in G and predicted_source != source_node:
        nx.draw_networkx_nodes(
            G, pos, nodelist=[predicted_source], ax=ax,
            node_color="#a855f7", node_size=260, node_shape="o",
            label=f"Predicted Source (Node {predicted_source})",
        )

    ax.set_title(title, color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    leg = ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")

    return fig


def export_cascade_to_json(
    data: Data,
    source_scores: Optional[Dict[int, float]] = None,
    bot_scores: Optional[Dict[int, float]] = None,
    veracity_label: Optional[str] = None,
) -> str:
    """
    Exports cascade graph to a JSON format consumed by D3.js or vis.js interactive frontend.
    """
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else data.x.size(0)
    G = build_networkx_tree(data.edge_index, num_nodes)
    levels = compute_tree_levels(G, root_node=0)

    nodes_data = []
    for node in G.nodes():
        src_score = source_scores.get(node, 0.0) if source_scores else (1.0 if node == 0 else 0.0)
        b_score = bot_scores.get(node, 0.0) if bot_scores else 0.0
        deg = G.out_degree(node)

        # Categorize
        if node == 0:
            role = "Root Source"
            color = "#fbbf24"
        elif b_score > 0.45 or deg >= 4:
            role = "Bot Hub"
            color = "#ef4444"
        else:
            role = "Retweeter"
            color = "#38bdf8"

        nodes_data.append({
            "id": int(node),
            "label": f"User {node}",
            "role": role,
            "color": color,
            "depth": int(levels.get(node, 0)),
            "outDegree": int(deg),
            "sourceProbability": round(float(src_score), 4),
            "botAnomalyScore": round(float(b_score), 4),
        })

    edges_data = []
    if data.edge_index.numel() > 0:
        for u, v in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):
            edges_data.append({"from": int(u), "to": int(v), "arrows": "to"})

    payload = {
        "veracity": veracity_label or ("Fake" if data.y.item() == 1 else "Real"),
        "totalNodes": num_nodes,
        "totalEdges": len(edges_data),
        "nodes": nodes_data,
        "edges": edges_data,
    }
    return json.dumps(payload, indent=2)
