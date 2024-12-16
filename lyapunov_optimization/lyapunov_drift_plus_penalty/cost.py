from typing import Dict, List
import numpy as np
from objects.task_object import task
from objects.vehicle_object import vehicle

def compute_v2v_transmission_cost(
    client_vehicle_index: int,
    client_vehicle_transmission_power: float,
    transmission_power_allocation_actions: Dict,
):
    return transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_index)][0] * \
        client_vehicle_transmission_power
    
def compute_v2i_transmission_cost(
    client_vehicle_index: int,
    client_vehicle_transmission_power: float,
    transmission_power_allocation_actions: Dict,
):
    return transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_index)][1] * \
        client_vehicle_transmission_power
        
def compute_i2i_transmission_cost(
    edge_node_index: int,
    distance_matrix_between_client_vehicles_and_edge_nodes: np.ndarray,
    distance_matrix_between_edge_nodes: np.ndarray,
    task_offloading_actions: Dict,
    client_vehicles: List[vehicle],
    tasks: List[task],
    maximum_client_vehicle_number: int,
    maximum_task_generation_number: int,
    now: int,
):
    cost = 0.0
    for i in range(maximum_client_vehicle_number):
        if distance_matrix_between_client_vehicles_and_edge_nodes[i][edge_node_index] == 1:
            tasks_of_vehicle_i = client_vehicles.get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), maximum_task_generation_number)
            if min_num > 0:
                for j in range(min_num):
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Edge Node") and \
                        int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1]) != edge_node_index:
                        other_edge_node_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        task_id = tasks_of_vehicle_i[j][1]
                        task_size = tasks[task_id].get_input_data_size()
                        distance = distance_matrix_between_edge_nodes[edge_node_index][other_edge_node_index]
                        cost += task_size * distance
    return cost


def compute_i2c_transmission_cost(
    edge_node_index: int,
    distance_matrix_between_client_vehicles_and_edge_nodes: np.ndarray,
    distance_matrix_between_edge_nodes_and_the_cloud: np.ndarray,
    task_offloading_actions: Dict,
    client_vehicles: List[vehicle],
    tasks: List[task],
    maximum_client_vehicle_number: int,
    maximum_task_generation_number: int,
    now: int,
):
    cost = 0.0
    for i in range(maximum_client_vehicle_number):
        if distance_matrix_between_client_vehicles_and_edge_nodes[i][edge_node_index] == 1:
            tasks_of_vehicle_i = client_vehicles.get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), maximum_task_generation_number)
            if min_num > 0:
                for j in range(min_num):
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Cloud":
                        task_id = tasks_of_vehicle_i[j][1]
                        task_size = tasks[task_id].get_input_data_size()
                        distance = distance_matrix_between_edge_nodes_and_the_cloud[edge_node_index]
                        cost += task_size * distance
    return cost

def compute_lc_computing_cost(
    client_vehicle_index: int,
    client_vehicle_computing_capability: float,
    task_offloaded_at_client_vehicles: Dict,
    computation_resource_allocation_actions: Dict,
):  
    cost = 0.0
    for task, index in enumerate(task_offloaded_at_client_vehicles["client_vehicle_" + str(client_vehicle_index)]):
        allocated_cycles = client_vehicle_computing_capability * \
            computation_resource_allocation_actions["client_vehicle_" + str(client_vehicle_index)][index]
        cost += allocated_cycles 
    return cost

def compute_vc_computing_cost(
    server_vehicle_index: int,
    server_vehicle_computing_capability: float,
    task_offloaded_at_server_vehicles: Dict,
    computation_resource_allocation_actions: Dict,
):
    cost = 0.0
    for task, index in enumerate(task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)]):
        allocated_cycles = server_vehicle_computing_capability * \
            computation_resource_allocation_actions["server_vehicle_" + str(server_vehicle_index)][index]
        cost += allocated_cycles
    return cost

def compute_ec_computing_cost(
    edge_node_index: int,
    edge_node_computing_capability: float,
    task_offloaded_at_edge_nodes: Dict,
    computation_resource_allocation_actions: Dict,
):
    cost = 0.0
    for task, index in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)]):
        allocated_cycles = edge_node_computing_capability * \
            computation_resource_allocation_actions["edge_node_" + str(edge_node_index)][index]
        cost += allocated_cycles
    return cost

def compute_cc_computing_cost(
    cloud_computing_capability: float,
    task_offloaded_at_cloud: Dict,
    computation_resource_allocation_actions: Dict,
):
    cost = 0.0
    for task, index in enumerate(task_offloaded_at_cloud["cloud"]):
        allocated_cycles = cloud_computing_capability * \
            computation_resource_allocation_actions["cloud"][index]
        cost += allocated_cycles
    return cost

# TODO: need to be normalized
def compute_total_cost(
    client_vehicle_number: int,
    v2v_transmission_costs : List[float],
    v2i_transmission_costs : List[float],
    lc_computing_costs : List[float],
    server_vehicle_number: int,
    vc_computing_costs : List[float],
    edge_node_number: int,
    ec_computing_costs : List[float],
    i2i_transmission_costs : List[float],
    i2c_transmission_costs : List[float],
    cc_computing_costs : List[float],
):
    total_cost = 0.0
    for i in range(client_vehicle_number):
        total_cost += v2v_transmission_costs[i] + v2i_transmission_costs[i] + lc_computing_costs[i]
    for i in range(server_vehicle_number):
        total_cost += vc_computing_costs[i]
    for i in range(edge_node_number):
        total_cost += ec_computing_costs[i] + i2i_transmission_costs[i] + i2c_transmission_costs[i]
    total_cost += cc_computing_costs[0]
    return total_cost