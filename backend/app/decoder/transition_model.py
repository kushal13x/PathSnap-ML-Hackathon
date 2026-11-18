"""Directed road-network graph routing and HMM transition probability model."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from backend.app.data.preprocessing import RoadNetworkMetric


@dataclass(frozen=True)
class TransitionConfig:
    """Parameters governing HMM transitions and temporal-gap segmentation."""

    gap_threshold_s: float = 30.0
    beta_m: float = 20.0
    uturn_penalty: float = 4.0
    backward_jitter_penalty: float = 1.5
    disconnected_penalty: float = 15.0
    max_route_search_m: float = 2500.0
    search_margin_m: float = 200.0


class DirectedRoadGraph:
    """Memory-light adjacency list with cached bounded Dijkstra shortest paths."""

    def __init__(self, road_net: RoadNetworkMetric) -> None:
        n_nodes = len(road_net.nx)
        adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_nodes)]
        u_arr = road_net.u
        v_arr = road_net.v
        len_arr = road_net.arc_lengths_m
        for arc_id in range(len(u_arr)):
            adj[int(u_arr[arc_id])].append(
                (int(v_arr[arc_id]), float(len_arr[arc_id]), arc_id)
            )
        self.adj = adj
        self.n_nodes = n_nodes
        # Cache: start_node -> (explored_budget_m, dist_dict, parent_dict)
        self._dijkstra_cache: dict[
            int, tuple[float, dict[int, float], dict[int, tuple[int, int]]]
        ] = {}

    def _run_bounded_dijkstra(
        self, start_node: int, budget_m: float
    ) -> tuple[dict[int, float], dict[int, tuple[int, int]]]:
        cached = self._dijkstra_cache.get(start_node)
        if cached is not None and cached[0] >= budget_m:
            return cached[1], cached[2]

        dist: dict[int, float] = {start_node: 0.0}
        parent: dict[int, tuple[int, int]] = {}
        pq: list[tuple[float, int]] = [(0.0, start_node)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist.get(u, float("inf")):
                continue
            if d_u > budget_m:
                break
            for v, w, arc_id in self.adj[u]:
                nd = d_u + w
                if nd <= budget_m and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    parent[v] = (u, arc_id)
                    heapq.heappush(pq, (nd, v))

        self._dijkstra_cache[start_node] = (budget_m, dist, parent)
        return dist, parent

    def shortest_distances_to_targets(
        self, start_node: int, target_nodes: np.ndarray, budget_m: float
    ) -> dict[int, float]:
        """Return shortest directed distances from start_node to target_nodes within budget_m."""
        # Fast path: if all targets already settled in cache with smaller budget, no re-search needed
        cached = self._dijkstra_cache.get(start_node)
        if cached is not None:
            cached_budget, dist_dict, _ = cached
            if cached_budget >= budget_m or all(int(t) in dist_dict for t in target_nodes):
                return {int(t): dist_dict.get(int(t), float("inf")) for t in target_nodes}

        dist_dict, _ = self._run_bounded_dijkstra(start_node, budget_m)
        return {int(t): dist_dict.get(int(t), float("inf")) for t in target_nodes}

    def shortest_path_arcs(
        self, start_node: int, end_node: int, budget_m: float
    ) -> list[int] | None:
        """Reconstruct valid directed arc ID sequence from start_node to end_node."""
        if start_node == end_node:
            return []
        dist_dict, parent_dict = self._run_bounded_dijkstra(start_node, budget_m)
        if end_node not in dist_dict:
            return None
        path_arcs: list[int] = []
        curr = end_node
        while curr != start_node:
            prev_node, arc_id = parent_dict[curr]
            path_arcs.append(arc_id)
            curr = prev_node
        path_arcs.reverse()
        return path_arcs


def compute_transition_log_probs(
    prev_arcs: np.ndarray,
    prev_proj_t: np.ndarray,
    curr_arcs: np.ndarray,
    curr_proj_t: np.ndarray,
    d_gps_m: float,
    road_net: RoadNetworkMetric,
    graph: DirectedRoadGraph,
    cfg: TransitionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (K_prev, K_curr) transition log-probabilities and connectivity flags."""
    k_prev = len(prev_arcs)
    k_curr = len(curr_arcs)
    log_probs = np.empty((k_prev, k_curr), dtype=np.float64)
    is_connected = np.zeros((k_prev, k_curr), dtype=bool)

    budget_m = min(
        cfg.max_route_search_m,
        max(350.0, 2.5 * d_gps_m + cfg.search_margin_m),
    )

    u_curr = road_net.u[curr_arcs]
    v_curr = road_net.v[curr_arcs]
    len_curr = road_net.arc_lengths_m[curr_arcs]
    uniq_u_curr = np.unique(u_curr)

    for i in range(k_prev):
        a_i = int(prev_arcs[i])
        u_i = int(road_net.u[a_i])
        v_i = int(road_net.v[a_i])
        len_i = float(road_net.arc_lengths_m[a_i])
        t_i = float(prev_proj_t[i])
        rem_i = (1.0 - t_i) * len_i

        # Only run bounded Dijkstra if needed for non-trivial multi-hop targets
        need_graph = any(
            int(curr_arcs[j]) != a_i and int(u_curr[j]) != v_i for j in range(k_curr)
        )
        dist_map = (
            graph.shortest_distances_to_targets(v_i, uniq_u_curr, budget_m)
            if need_graph
            else {}
        )

        for j in range(k_curr):
            a_j = int(curr_arcs[j])
            u_j = int(u_curr[j])
            v_j = int(v_curr[j])
            len_j = float(len_curr[j])
            t_j = float(curr_proj_t[j])

            if a_i == a_j:
                # Stay on same road arc
                is_connected[i, j] = True
                d_route = max(0.0, (t_j - t_i) * len_i)
                pen = (
                    cfg.backward_jitter_penalty
                    if (t_j < t_i - 0.15 and d_gps_m > 10.0)
                    else 0.0
                )
                log_probs[i, j] = -abs(d_route - d_gps_m) / cfg.beta_m - pen
            elif v_i == u_j:
                # Direct 1-hop road connection
                is_connected[i, j] = True
                d_route = rem_i + t_j * len_j
                is_uturn = u_i == v_j
                pen = cfg.uturn_penalty if is_uturn else 0.0
                log_probs[i, j] = -abs(d_route - d_gps_m) / cfg.beta_m - pen
            else:
                g_dist = dist_map.get(u_j, float("inf"))
                if np.isfinite(g_dist):
                    is_connected[i, j] = True
                    d_route = rem_i + g_dist + t_j * len_j
                    is_uturn = u_i == v_j
                    pen = cfg.uturn_penalty if is_uturn else 0.0
                    log_probs[i, j] = -abs(d_route - d_gps_m) / cfg.beta_m - pen
                else:
                    # Disconnected or beyond search budget: finite fallback penalty
                    is_connected[i, j] = False
                    jump_gap = float(
                        np.hypot(
                            road_net.nx[u_j] - road_net.nx[v_i],
                            road_net.ny[u_j] - road_net.ny[v_i],
                        )
                    )
                    d_fallback = rem_i + jump_gap + t_j * len_j
                    log_probs[i, j] = (
                        -abs(d_fallback - d_gps_m) / cfg.beta_m
                        - cfg.disconnected_penalty
                    )

    return log_probs, is_connected
