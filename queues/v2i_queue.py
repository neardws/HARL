from queues.base_queue import baseQueue
from objects.task_object import task
from objects.vehicle_object import vehicle
from typing import List, Dict
import numpy as np
from utilities.time_calculation import compute_transmission_rate, compute_V2I_SINR
from utilities.conversion import cover_MB_to_bit

class V2IQueue(baseQueue):
    def __init__(
        self,
        time_slot_num: int,
        name: str,
        edge_node_index : int,
        client_vehicles: List[vehicle],
        client_vehicle_number: int,
        maximum_task_generation_number: int,
        white_gaussian_noise,
        V2I_bandwidth: float,
        channel_gains_between_client_vehicle_and_edge_nodes: np.ndarray,
    ):
        self._edge_node_index = edge_node_index
        self._client_vehicles = client_vehicles
        self._client_vehicle_number = client_vehicle_number
        self._maximum_task_generation_number = maximum_task_generation_number
        self._white_gaussian_noise = white_gaussian_noise
        self._V2I_bandwidth = V2I_bandwidth
        self._channel_gains_between_client_vehicle_and_edge_nodes = channel_gains_between_client_vehicle_and_edge_nodes
        super().__init__(time_slot_num=time_slot_num, name=name)
        
    def compute_input(
        self,
        task_uploaded_at_edge_nodes,
    ):
        input = 0.0
        for index, task in enumerate(task_uploaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            task_index = task["task_index"]
            task_size = cover_MB_to_bit(task["task"].get_input_data_size())
            task_arrival_rate = self._client_vehicles[task["client_vehicle_index"]].get_task_arrival_rate_by_task_index(task_index)
            input += task_size * task_arrival_rate
        return input
    
    def compute_output(
        self,
        task_uploaded_at_edge_nodes,
        transmission_power_allocation_actions: Dict,
        now: int,
    ):
        output = 0.0
        client_vehicle_indexs = []
        for index, task in enumerate(task_uploaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            if task["client_vehicle_index"] not in client_vehicle_indexs:
                client_vehicle_indexs.append(task["client_vehicle_index"])
                if task["edge_index"] == self._edge_node_index:         # the client vehicle in under the coverage of the edge node
                    transmission_rate = self.obtain_transmission_rate(
                        task_uploaded_at_edge_nodes=task_uploaded_at_edge_nodes,
                        transmission_power_allocation_actions=transmission_power_allocation_actions,
                        client_vehicle_index=task["client_vehicle_index"],
                        edge_node_index=self._edge_node_index,
                        now=now,
                    )
                    output += transmission_rate
        return output
    
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
         