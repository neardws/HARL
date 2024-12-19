import random
from typing import List, Dict, Tuple
from objects.mobility_object import mobility
from objects.vehicle_object import vehicle
from objects.edge_node_object import edge_node
from objects.cloud_server_object import cloud_server
from utilities.vehicular_trajectories_processing import TrajectoriesProcessing
from utilities.wired_bandwidth import get_wired_bandwidth_between_edge_nodes_and_the_cloud

def generate_task_set(
    task_num: int,
    task_seeds: List[int],
    distribution: str,
    min_input_data_size: float,     # in MB
    max_input_data_size: float,     # in MB
    min_cqu_cycles: float,          # cycles/bit
    max_cqu_cycles: float,          # cycles/bit
) -> List[Dict]:
    tasks = []
    if distribution == "uniform":
        for _ in range(task_num):
            random.seed(task_seeds[_])
            # 生成两个随机值
            value1 = random.uniform(min_input_data_size, max_input_data_size)
            value2 = random.uniform(min_input_data_size, max_input_data_size)

            # 比较两个值，确定 t_min_input_data_size 和 t_max_input_data_size
            t_min_input_data_size = min(value1, value2)
            t_max_input_data_size = max(value1, value2)
            
            t = {
                "task_index": _,
                "min_input_data_size": t_min_input_data_size,
                "max_input_data_size": t_max_input_data_size,
                "cqu_cycles": random.uniform(min_cqu_cycles, max_cqu_cycles),
            }
            tasks.append(t)
        return tasks
    elif distribution == "normal":
        raise ValueError("Distribution not supported")
    else:
        raise ValueError("Distribution not supported")
    
    
def generate_vehicles(
    vehicle_num: int,
    vehicle_seeds: List[int],
    slot_length: int,
    file_name_key: str,
    slection_way: str,
    filling_way: str,
    chunk_size: int,
    start_time: str,
    min_computing_capability: float,        # GHz
    max_computing_capability: float,        # GHz
    min_storage_capability: float,          # MB
    max_storage_capability: float,          # MB
    min_transmission_power: float,          # mW
    max_transmission_power: float,          # mW
    communication_range: float,             # meters
    min_task_arrival_rate: float,           # tasks/s
    max_task_arrival_rate: float,           # tasks/s
    server_vehicle_probability: float,
    tasks,
    task_num: int,
    task_ids_rate: float,
    distribution: str,
) -> Tuple[float, float, float, float, List[vehicle]]:

    trajectoriesProcessing = TrajectoriesProcessing(
        file_name_key=file_name_key,
        vehicle_number=vehicle_num,
        start_time=start_time,
        slot_length=slot_length,
        slection_way=slection_way,
        filling_way=filling_way,
        chunk_size=chunk_size,
    )
    min_map_x, max_map_x, min_map_y, max_map_y = trajectoriesProcessing.processing()
    
    mobilities_list : List[List[mobility]] = trajectoriesProcessing.get_vehicle_mobilities()
    
    vehicles = []
    if distribution == "uniform":
        for _ in range(vehicle_num):
            random.seed(vehicle_seeds[_])
            vehicles.append(
                vehicle(
                    random_seed=vehicle_seeds[_],
                    mobilities=mobilities_list[_],
                    
                    computing_capability=random.uniform(min_computing_capability, max_computing_capability),
                    storage_capability=random.uniform(min_storage_capability, max_storage_capability),
                    time_slot_num=slot_length,
                    
                    transmission_power=random.uniform(min_transmission_power, max_transmission_power),
                    communication_range=communication_range,
                    
                    min_task_arrival_rate=min_task_arrival_rate,
                    max_task_arrival_rate=max_task_arrival_rate,
                    
                    server_vehicle_probability = server_vehicle_probability,
                    
                    task_num=task_num,
                    task_ids_rate=task_ids_rate,
                    tasks=tasks,
                )
            )
        return min_map_x, max_map_x, min_map_y, max_map_y, vehicles
    else:
        raise Exception("distribution not supported")
    
    
def generate_edge_nodes(
    edge_num: int,
    edge_seeds: List[int],
    min_map_x: float,
    max_map_x: float,
    min_map_y: float,
    max_map_y: float,
    min_computing_capability: float,        # GHz   
    max_computing_capability: float,        # GHz
    min_storage_capability: float,          # MB
    max_storage_capability: float,          # MB
    communication_range : float,            # meters
    time_slot_num: int,
    distribution: str,
) -> List[edge_node]:
    edge_nodes = []
    
    if distribution == "uniform" :
        for _ in range(edge_num):
            random.seed(edge_seeds[_])
            mobility_obj = mobility(
                x = random.uniform(min_map_x, max_map_x), 
                y = random.uniform(min_map_y, max_map_y), 
                speed = 0, 
                acceleration = 0,
                direction = 0,
                time = 0
            )
            computing_capability_list = random.random()
            storage_capability = random.random() 
            if computing_capability_list < 0.3:
                computing_capability_list = min_computing_capability
                storage_capability = min_storage_capability
            elif computing_capability_list > 0.7:
                computing_capability_list = max_computing_capability
                storage_capability = max_storage_capability
            else:
                computing_capability_list = (min_computing_capability + max_computing_capability) / 2
                storage_capability = (min_storage_capability + max_storage_capability) / 2
            edge_node_obj = edge_node(
                edge_node_mobility = mobility_obj, 
                computing_capability = computing_capability_list, 
                storage_capability = storage_capability,
                communication_range = communication_range,
                time_slot_num = time_slot_num)
            edge_nodes.append(edge_node_obj)
        return edge_nodes 
    else:
        raise Exception("distribution is not supported")
        
        
def generate_cloud(
    cloud_seed: int,
    min_map_x: float,
    max_map_x: float,
    min_map_y: float,
    max_map_y: float,
    computing_capability: float,        # GHz
    storage_capability: float,          # MB
    edge_node_num: int,
    time_slot_num: int,
    min_wired_bandwidth: float,         # Mbps
    max_wired_bandwidth: float,         # Mbps
    distribution: str,
) -> cloud_server:
    if distribution == "uniform":
        random.seed(cloud_seed)
        wired_bandwidths = get_wired_bandwidth_between_edge_nodes_and_the_cloud(
            min_wired_bandwidth=min_wired_bandwidth,
            max_wired_bandwidth=max_wired_bandwidth,
            edge_node_num=edge_node_num,
        )
        cloud_mobility = mobility(
            x = min_map_x + (max_map_x - min_map_x) / 2, 
            y = min_map_y + (max_map_y - min_map_y) / 2, 
            speed = 0, 
            acceleration = 0,
            direction = 0,
            time = 0
        )
        return cloud_server(cloud_mobility, computing_capability, storage_capability, time_slot_num, wired_bandwidths)
    else:
        raise Exception("No such distribution: " + distribution + " for cloud")
    
