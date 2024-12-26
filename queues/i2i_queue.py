from queues.base_queue import baseQueue
from objects.vehicle_object import vehicle
from typing import List, Dict
import numpy as np
from utilities.time_calculation import compute_transmission_rate, compute_V2I_SINR
from utilities.conversion import cover_Mbps_to_bps, cover_kms_to_ms

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
        super().__init__(time_slot_num=time_slot_num, name=name)
        
    def compute_input(
        self,
        task_offloaded_at_edge_nodes,
        task_uploaded_at_edge_nodes,
        transmission_power_allocation_actions: Dict,
        now: int,
    ):
        input = 0.0
        client_vehicle_indexs = []
        for index, task in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            client_vehicle_index = task["client_vehicle_index"]
            if client_vehicle_index not in client_vehicle_indexs:
                client_vehicle_indexs.append(client_vehicle_index)
                edge_node_index = task["edge_index"]
                if edge_node_index != self._edge_node_index:
                    transmission_rate = self.obtain_transmission_rate(
                        task_uploaded_at_edge_nodes=task_uploaded_at_edge_nodes,
                        transmission_power_allocation_actions=transmission_power_allocation_actions,
                        client_vehicle_index=client_vehicle_index,
                        edge_node_index=edge_node_index,
                        now=now,
                    )
                    input += transmission_rate
        return input
    
    def obtain_transmission_rate(
        self,
        task_uploaded_at_edge_nodes,
        transmission_power_allocation_actions: Dict,
        client_vehicle_index: int,
        edge_node_index: int,
        now: int,
    ):
        interference = 0.0
        channel_gain = self._channel_gains_between_client_vehicle_and_edge_nodes[client_vehicle_index][edge_node_index][now]
        client_gain = np.abs(channel_gain) ** 2
        other_client_vehicle_indexs = []
        for index, task in enumerate(task_uploaded_at_edge_nodes["edge_node_" + str(edge_node_index)]):
            other_client_vehicle_index = task["client_vehicle_index"]
            if other_client_vehicle_index != client_vehicle_index and other_client_vehicle_index not in other_client_vehicle_indexs:
                other_client_vehicle_indexs.append(other_client_vehicle_index)
                other_gain = np.abs(self._channel_gains_between_client_vehicle_and_edge_nodes[other_client_vehicle_index][edge_node_index][now]) ** 2
                if other_gain < client_gain:
                    interference += other_gain * self._client_vehicles[other_client_vehicle_index].get_transmission_power() * \
                        transmission_power_allocation_actions["client_vehicle_" + str(other_client_vehicle_index)][1]
        
        sinr = compute_V2I_SINR(
            white_gaussian_noise=self._white_gaussian_noise,
            channel_gain=channel_gain,
            transmission_power=self._client_vehicles[client_vehicle_index].get_transmission_power() * transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_index)][1],
            interference=interference,
        )
        transmission_rate = compute_transmission_rate(SINR=sinr, bandwidth=self._V2I_bandwidth)
        return transmission_rate
    
    def compute_output(
        self,
        task_offloaded_at_edge_nodes,
        distance_matrix_between_edge_nodes: np.ndarray,
    ):
        output = 0.0
        for index, task in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            edge_node_index = task["edge_index"]
            if edge_node_index != self._edge_node_index:
                transmission_rate = cover_Mbps_to_bps(self._I2I_transmission_rate)
                propagation_speed = cover_kms_to_ms(self._I2I_propagation_speed)
                distance = distance_matrix_between_edge_nodes[self._edge_node_index][edge_node_index]
                depature_rate = self.obtain_departure_rate(
                    transmission_rate=transmission_rate,
                    propagation_speed=propagation_speed,
                    distance=distance,
                )
                output += depature_rate
        return output

    
    def obtain_departure_rate(
        self,
        transmission_rate: float,
        propagation_speed: float,
        distance: float,
    ):
        # 传播延迟
        propagation_delay = distance / propagation_speed
        
        # 有效吞吐量
        departure_rate = transmission_rate / (1 + propagation_delay * transmission_rate)
        
        return departure_rate
        