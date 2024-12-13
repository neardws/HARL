from typing import List
import numpy as np
from objects.mobility_object import mobility
from objects.vehicle_object import vehicle
from objects.edge_node_object import edge_node

def calculate_distance(
    mobility1: mobility, 
    mobility2: mobility,
    type: str,
) -> float:
    if type == "vehicles":
        if mobility1.get_time() != mobility2.get_time():
            raise ValueError("The time of two mobilities are not the same.")
        return ((mobility1.get_x() - mobility2.get_x()) ** 2 + (mobility1.get_y() - mobility2.get_y()) ** 2) ** 0.5
    elif type == "edge_nodes":
        return ((mobility1.get_x() - mobility2.get_x()) ** 2 + (mobility1.get_y() - mobility2.get_y()) ** 2) ** 0.5
    else:
        raise Exception("type not supported")

def get_distance_matrix_between_client_vehicles_and_server_vehicles(
    client_vehicles: List[vehicle],
    server_vehicles: List[vehicle],
    now: int
) -> np.ndarray:
    client_vehicle_num = len(client_vehicles)
    server_vehicle_num = len(server_vehicles)
    distance_matrix = np.zeros((client_vehicle_num, server_vehicle_num))
    for i in range(client_vehicle_num):
        for j in range(server_vehicle_num):
            if i != j:
                distance_matrix[i][j] = calculate_distance(
                    client_vehicles[i].get_mobility(now), 
                    server_vehicles[j].get_mobility(now),
                    type="vehicles"
                )
    return distance_matrix

def get_vehicles_under_V2V_communication_range(
    distance_matrix : np.ndarray,
    client_vehicles: List[vehicle],
    server_vehicles: List[vehicle],
) -> np.ndarray:
    vehicles_under_V2V_communication_range = np.zeros(
        (len(client_vehicles), len(server_vehicles)))
    for i in range(len(client_vehicles)):
        for j in range(len(server_vehicles)):
            if i != j:
                if distance_matrix[i][j] <= client_vehicles[i].get_communication_range():
                    vehicles_under_V2V_communication_range[i][j] = 1
    return vehicles_under_V2V_communication_range



def get_distance_matrix_between_vehicles_and_edge_nodes(
    client_vehicles: List[vehicle],
    edge_nodes: List[edge_node],
    now: int,
) -> np.ndarray:
    vehicle_num = len(client_vehicles)
    edge_node_num = len(edge_nodes)
    distance_matrix = np.zeros((vehicle_num, edge_node_num))
    for i in range(vehicle_num):
        for j in range(edge_node_num):
            distance_matrix[i][j] = calculate_distance(
                client_vehicles[i].get_mobility(now = now), 
                edge_nodes[j].get_mobility(),
                type="edge_nodes"
            )
    return distance_matrix

def get_vehicles_under_V2I_communication_range(
    client_vehicles: List[vehicle],
    edge_nodes: List[edge_node],
    now: int,
) -> np.ndarray:
    num_vehicles = len(client_vehicles)
    num_edge_nodes = len(edge_nodes)
    vehicles_under_V2I_communication_range = np.zeros((num_vehicles, num_edge_nodes))
    for i in range(num_vehicles):
        for j in range(num_edge_nodes):
            distance = calculate_distance(
                client_vehicles[i].get_mobility(now), 
                edge_nodes[j].get_mobility(),
                type="edge_nodes"
            )
            if(distance <= edge_nodes[j].get_communication_range()):
                vehicles_under_V2I_communication_range[i][j] = 1
    return vehicles_under_V2I_communication_range


def get_distance_matrix_between_edge_nodes(
    edge_nodes: List[edge_node],
) -> np.ndarray:
    num_edge_nodes = len(edge_nodes)
    distance_matrix = np.zeros((num_edge_nodes, num_edge_nodes))
    for i in range(num_edge_nodes):
        for j in range(num_edge_nodes):
            distance_matrix[i][j] = calculate_distance(
                edge_nodes[i].get_mobility(), 
                edge_nodes[j].get_mobility(),
                type="edge_nodes"
            )
    return distance_matrix



