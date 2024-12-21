from typing import List
import numpy as np
from objects.vehicle_object import vehicle
from objects.edge_node_object import edge_node

def generate_rayleigh_distributed_small_scale_fading(size: int = 1):
    return np.random.rayleigh(scale=1, size=size)

def compute_channel_gain(
    rayleigh_distributed_small_scale_fading: np.ndarray,
    distance: float,
    path_loss_exponent: int,
) -> np.ndarray:
    return rayleigh_distributed_small_scale_fading / np.power(distance, path_loss_exponent / 2)

def obtain_channel_gains_between_client_vehicle_and_server_vehicles(
    client_vehicles: List[vehicle],
    server_vehicles: List[vehicle],
    time_slot_num: int,
    distance_matrix: np.ndarray,
    path_loss_exponent: int,
) -> np.ndarray:
    channel_gains = np.zeros((len(client_vehicles), len(server_vehicles), time_slot_num))
    for t in range(time_slot_num):
        for i in range(len(client_vehicles)):
            for j in range(len(server_vehicles)):
                distance = distance_matrix[i][j][t]
                fading = generate_rayleigh_distributed_small_scale_fading()
                channel_gain = compute_channel_gain(fading, distance, path_loss_exponent)
                channel_gains[i][j][t] = channel_gain
    return channel_gains

def obtain_channel_gains_between_vehicles_and_edge_nodes(
    client_vehicles: List[vehicle],
    edge_nodes: List[edge_node],
    time_slot_num: int,
    distance_matrix: np.ndarray,
    path_loss_exponent: int,
) -> np.ndarray:
    channel_gains = np.zeros((len(client_vehicles), len(edge_nodes), time_slot_num))
    for t in range(time_slot_num):
        for i in range(len(client_vehicles)):
            for j in range(len(edge_nodes)):
                distance = distance_matrix[i][j][t]
                fading = generate_rayleigh_distributed_small_scale_fading()
                channel_gain = compute_channel_gain(fading, distance, path_loss_exponent)
                channel_gains[i][j][t] = channel_gain
    return channel_gains
