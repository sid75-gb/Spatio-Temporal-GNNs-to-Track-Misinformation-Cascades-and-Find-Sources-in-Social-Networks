"""
Structural bot network detection and coordinated inauthentic amplifier isolation.
Identifies amplification hubs and coordinated bot rings in misinformation cascades.
"""
from typing import Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


class BotNetworkIsolator:
    """
    Isolates structural bot hubs and coordinated inauthentic amplification rings
    driving misinformation diffusion trees.
    """

    def __init__(
        self,
        velocity_threshold: float = 1.5,
        min_fanout: int = 4,
        profile_anomaly_weight: float = 0.35,
        velocity_weight: float = 0.35,
        fanout_weight: float = 0.30,
    ):
        self.velocity_threshold = velocity_threshold
        self.min_fanout = min_fanout
        self.w_profile = profile_anomaly_weight
        self.w_velocity = velocity_weight
        self.w_fanout = fanout_weight

    def extract_node_anomaly_scores(
        self,
        data: Data,
        G: nx.DiGraph,
        temporal_velocity: Optional[torch.Tensor] = None,
    ) -> Dict[int, float]:
        """
        Computes composite inauthentic amplification score for every node in cascade:
        Score = w_fanout * FanOutScore + w_velocity * VelocityScore + w_profile * ProfileAnomaly
        """
        n = G.number_of_nodes()
        if n == 0:
            return {}

        out_degrees = dict(G.out_degree())
        max_deg = max(out_degrees.values()) if out_degrees else 1

        # 1. Structural Fan-out score (bots act as high-degree broadcast re-seeders)
        fanout_scores = {node: out_degrees.get(node, 0) / (max_deg + 1e-6) for node in G.nodes()}

        # 2. Temporal velocity score (bursty immediate propagation)
        if temporal_velocity is not None and temporal_velocity.numel() == n:
            vel_np = temporal_velocity.cpu().numpy()
            vel_min, vel_max = vel_np.min(), vel_np.max()
            if vel_max > vel_min:
                vel_norm = (vel_np - vel_min) / (vel_max - vel_min)
            else:
                vel_norm = np.zeros(n)
            velocity_scores = {node: float(vel_norm[node]) for node in range(n)}
        else:
            # Fallback: nodes closer to early diffusion with high fanout have higher velocity
            levels = nx.single_source_shortest_path_length(G.to_undirected(), 0) if 0 in G else {}
            velocity_scores = {}
            for node in G.nodes():
                lvl = levels.get(node, 5)
                # Lower level (earlier) + higher fanout = higher velocity
                velocity_scores[node] = float(np.exp(-0.5 * lvl) * (out_degrees.get(node, 0) / (max_deg + 1e-6)))

        # 3. Profile feature anomaly (if profile features present in node features)
        # UPFD profile features: 10 dimensions (followers, friends, verified, etc.)
        profile_scores = {node: 0.0 for node in G.nodes()}
        if data.x is not None and data.x.size(-1) >= 10:
            # Check last 10 dimensions for profile cues
            prof_x = data.x[:, -10:].cpu().numpy()
            # Suspicious if followers-to-friends ratio is extremely abnormal or low
            followers = np.abs(prof_x[:, 0])
            friends = np.abs(prof_x[:, 1]) + 1e-6
            ff_ratio = followers / friends
            # Extreme outliers in profile attributes
            prof_zscore = np.abs(prof_x - np.mean(prof_x, axis=0)) / (np.std(prof_x, axis=0) + 1e-6)
            mean_anomaly = np.mean(prof_zscore, axis=1)
            prof_min, prof_max = mean_anomaly.min(), mean_anomaly.max()
            if prof_max > prof_min:
                norm_prof = (mean_anomaly - prof_min) / (prof_max - prof_min)
                profile_scores = {node: float(norm_prof[node]) for node in range(n)}

        # Composite score
        composite_scores = {}
        for node in G.nodes():
            score = (
                self.w_fanout * fanout_scores.get(node, 0.0)
                + self.w_velocity * velocity_scores.get(node, 0.0)
                + self.w_profile * profile_scores.get(node, 0.0)
            )
            composite_scores[node] = float(score)

        return composite_scores

    def isolate_bot_rings(
        self,
        data: Data,
        G: nx.DiGraph,
        threshold: float = 0.5,
    ) -> Dict[str, Union[List[int], List[List[int]], Dict[int, float]]]:
        """
        Isolates individual suspicious bot nodes and partitions them into coordinated rings.
        """
        scores = self.extract_node_anomaly_scores(data, G)

        # Filter suspicious amplifier nodes (excluding root post 0)
        suspicious_nodes = [
            node for node, score in scores.items()
            if node != 0 and (score >= threshold or G.out_degree(node) >= self.min_fanout)
        ]

        # Identify coordinated rings (clusters among suspicious nodes)
        subgraph = G.to_undirected().subgraph(suspicious_nodes)
        coordinated_rings = [list(comp) for comp in nx.connected_components(subgraph)]

        return {
            "suspicious_nodes": suspicious_nodes,
            "coordinated_rings": coordinated_rings,
            "anomaly_scores": scores,
            "num_isolated_bots": len(suspicious_nodes),
            "num_rings": len(coordinated_rings),
        }
