from queues.base_queue import baseQueue
from objects.task_object import task
from objects.vehicle_object import vehicle
from typing import List, Dict
import numpy as np
from utilities.time_calculation import compute_transmission_rate, compute_V2I_SINR


class I2IQueue(baseQueue):
    def __init__(
        self,
        time_slot_num: int,
        name: str,
        client_vehicles: List[vehicle],
        client_vehicle_number: int,
        maximum_task_generation_number: int,
        edge_node_number: int,
        edge_node_index: int,
        white_gaussian_noise,
        V2I_bandwidth: float,
        I2I_transmission_rate: float,
        I2I_propagation_speed: float,
        channel_gains_between_client_vehicle_and_edge_nodes: np.ndarray,
    ):
        self._client_vehicles = client_vehicles
        self._client_vehicle_number = client_vehicle_number
        self._maximum_task_generation_number = maximum_task_generation_number
        self._edge_node_number = edge_node_number
        self._edge_node_index = edge_node_index
        self._white_gaussian_noise = white_gaussian_noise
        self._V2I_bandwidth = V2I_bandwidth
        self._I2I_transmission_rate = I2I_transmission_rate
        self._I2I_propagation_speed = I2I_propagation_speed
        self._channel_gains_between_client_vehicle_and_edge_nodes = channel_gains_between_client_vehicle_and_edge_nodes
        super(I2IQueue, self).__init__(time_slot_num=time_slot_num, name=name)
        
    def compute_input(
        self,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        vehicles_under_V2I_communication_range: np.ndarray,
        now: int,
    ):
        # 根据task_offloading_actions筛选，进一步筛选V2I通信距离内，最后求和
        input = 0.0
        for e in range(self._edge_node_number):
            if e != self._edge_node_index:
                for i in range(self._client_vehicle_number):
                    if vehicles_under_V2I_communication_range[i][e][now] == 1:
                        tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
                        min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
                        if min_num > 0:
                            tag = False
                            for j in range(min_num):
                                if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Edge Node " + str(self._edge_node_index):
                                    tag = True
                                    break
                            if tag:
                                transmission_rate = self.obtain_transmission_rate(
                                    edge_node_index=e,
                                    task_offloading_actions=task_offloading_actions,
                                    transmission_power_allocation_actions=transmission_power_allocation_actions,
                                    vehicles_under_V2I_communication_range=vehicles_under_V2I_communication_range,
                                    now=now,
                                    client_vehicle_index=i,
                                )
                                input += transmission_rate 
        return input
    
    def obtain_transmission_rate(
        self,
        edge_node_index: int,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        vehicles_under_V2I_communication_range: np.ndarray,
        now: int,
        client_vehicle_index: int,
    ):
        # 计算SINR
        interference = 0.0
        for i in range(self._client_vehicle_number):
            if vehicles_under_V2I_communication_range[i][edge_node_index][now] == 1:
                tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
                min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
                if min_num > 0:
                    tag = False
                    for j in range(min_num):
                        if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] .startswith("Edge Node") or \
                            task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Cloud":
                            tag = True
                            break
                    if tag:
                        interference += np.abs(self._channel_gains_between_client_vehicle_and_edge_nodes[i][edge_node_index][now]) ** 2 * \
                            self._client_vehicles[i].get_transmission_power() * transmission_power_allocation_actions["client_vehicle_" + str(i)][1]
        
        sinr = compute_V2I_SINR(
            white_gaussian_noise=self._white_gaussian_noise,
            channel_gain=self._channel_gains_between_client_vehicle_and_edge_nodes[client_vehicle_index][edge_node_index][now],
            transmission_power=self._client_vehicles[client_vehicle_index].get_transmission_power() * transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_index)][1],
            interference=interference,
        )
        
        transmission_rate = compute_transmission_rate(SINR=sinr, bandwidth=self._V2I_bandwidth)
        return transmission_rate
    
    def compute_output(
        self,
        task_offloading_actions: Dict,
        vehicles_under_V2I_communication_range: np.ndarray,
        distance_matrix_between_edge_nodes: np.ndarray,
        now: int,
    ):
        output = 0.0
        for e in range(self._edge_node_number):
            if e != self._edge_node_index:
                for i in range(self._client_vehicle_number):
                    if vehicles_under_V2I_communication_range[i][e][now] == 1:
                        tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
                        min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
                        if min_num > 0:
                            for j in range(min_num):
                                if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Edge Node " + str(e):
                                    task_id = tasks_of_vehicle_i[j][1]
                                    task_data_size = tasks_of_vehicle_i[j][2].get_input_data_size()
                                    transmission_rate = self._I2I_transmission_rate
                                    propagation_speed = self._I2I_propagation_speed
                                    distance = distance_matrix_between_edge_nodes[e][self._edge_node_index]
                                    depature_rate = self.obtain_departure_rate(
                                        task_data_size=task_data_size,
                                        transmission_rate=transmission_rate,
                                        propagation_speed=propagation_speed,
                                        distance=distance,
                                    )
                                    output += depature_rate 
        return output    
    
    def obtain_departure_rate(
        self,
        task_data_size: float,
        transmission_rate: float,
        propagation_speed: float,
        distance: float,
    ):
        return task_data_size * transmission_rate * propagation_speed / \
            (task_data_size * propagation_speed + distance * transmission_rate)