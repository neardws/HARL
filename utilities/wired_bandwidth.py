from typing import List
from Objects.edge_node_object import edge_node
import numpy as np
import random

def get_wired_bandwidth_between_edge_node_and_other_edge_nodes(
    edge_nodes: List[edge_node],
    weight: float,
    transmission_rate: float,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    num_edge_nodes = len(edge_nodes)
    result = np.zeros((num_edge_nodes, num_edge_nodes))
    for i in range(num_edge_nodes):
        for j in range(num_edge_nodes):
            if i == j:
                continue
            result[i, j] = weight * transmission_rate / distance_matrix[i][j]
    return result

def get_wired_bandwidth_between_edge_nodes_and_the_cloud(
    min_wired_bandwidth: float,
    max_wired_bandwidth: float,
    edge_node_num: int,
) -> List[float]:
    wired_bandwidths = [random.uniform(min_wired_bandwidth, max_wired_bandwidth) for _ in range(edge_node_num)]
    return wired_bandwidths