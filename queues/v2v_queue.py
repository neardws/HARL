from queues.base_queue import baseQueue
from objects.task_object import task
from objects.vehicle_object import vehicle
from typing import List, Dict
import numpy as np
from utilities.time_calculation import compute_transmission_rate, compute_V2V_SINR

class V2VQueue(baseQueue):
    def __init__(
        self,
        time_slot_num: int,
        name: str,
        server_vehicle_index : int,
        client_vehicles: List[vehicle],
        client_vehicle_number: int,
        maximum_task_generation_number: int,
        white_gaussian_noise,
        V2V_bandwidth: float,
        channel_gains_between_client_vehicle_and_server_vehicles: np.ndarray,
    ):
        self._server_vehicle_index = server_vehicle_index
        self._client_vehicles = client_vehicles
        self._client_vehicle_number = client_vehicle_number
        self._maximum_task_generation_number = maximum_task_generation_number
        self._white_gaussian_noise = white_gaussian_noise
        self._V2V_bandwidth = V2V_bandwidth
        self._channel_gains_between_client_vehicle_and_server_vehicles = channel_gains_between_client_vehicle_and_server_vehicles
        super(V2VQueue, self).__init__(time_slot_num=time_slot_num, name=name)

    def compute_input(
        self,
        task_offloading_actions: Dict,
        vehicles_under_V2V_communication_range: np.ndarray,
        now: int,
    ):
        # 根据task_offloading_actions筛选，进一步筛选V2V通信距离内，最后求和
        input = 0.0
        for i in range(self._client_vehicle_number):
            tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
            if min_num > 0:
                for j in range(min_num):
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == \
                        "Server Vehicle " + str(self._server_vehicle_index) and \
                        vehicles_under_V2V_communication_range[i][self._server_vehicle_index] == 1:
                        task_index = tasks_of_vehicle_i[j][1]
                        task_size = tasks_of_vehicle_i[j][2].get_input_data_size()
                        task_arrival_rate = self._client_vehicles[i].get_task_arrival_rate_by_task_index(task_index)
                        input += task_size * task_arrival_rate
        return input
    
    def compute_output(
        self,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        vehicles_under_V2V_communication_range: np.ndarray,
        now: int,
    ):
        output = 0.0
        for i in range(self._client_vehicle_number):
            tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
            if min_num > 0:
                tag = False
                for j in range(min_num):
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == \
                        "Server Vehicle " + str(self._server_vehicle_index) and \
                        vehicles_under_V2V_communication_range[i][self._server_vehicle_index] == 1:
                        tag = True
                        break
                if tag:
                    transmission_rate = self.obtain_transmission_rate(
                        task_offloading_actions=task_offloading_actions,
                        transmission_power_allocation_actions=transmission_power_allocation_actions,
                        vehicles_under_V2V_communication_range=vehicles_under_V2V_communication_range,
                        now=now,
                        client_vehicle_index=i,
                        server_vehicle_index=self._server_vehicle_index,
                    )
                    outpu += transmission_rate 
        return output
                    
    def obtain_transmission_rate(
        self,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        vehicles_under_V2V_communication_range: np.ndarray,
        now: int,
        client_vehicle_index: int,
        server_vehicle_index: int,
    ):
        # 计算SINR
        interference = 0.0
        for i in range(self._client_vehicle_number):
            tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
            if min_num > 0:
                tag = False
                for j in range(min_num):
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == \
                        "Server Vehicle " + str(server_vehicle_index) and \
                        vehicles_under_V2V_communication_range[i][self._server_vehicle_index] == 1:
                        tag = True
                        break
                if tag:
                    if i != client_vehicle_index:
                        if np.abs(self._channel_gains_between_client_vehicle_and_server_vehicles[i][server_vehicle_index]) ** 2 < \
                            np.abs(self._channel_gains_between_client_vehicle_and_server_vehicles[client_vehicle_index][server_vehicle_index]) ** 2:
                            interference += np.abs(self._channel_gains_between_client_vehicle_and_server_vehicles[i][self._server_vehicle_index]) ** 2 * \
                                self._client_vehicles[i].get_transmission_power() * transmission_power_allocation_actions["client_vehicle_" + str(i)][0]
        
        sinr = compute_V2V_SINR(
            white_gaussian_noise=self._white_gaussian_noise,
            channel_gain=self._channel_gains_between_client_vehicle_and_server_vehicles[client_vehicle_index][server_vehicle_index],
            transmission_power=self._client_vehicles[client_vehicle_index].get_transmission_power()  * transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_index)][0],
            interference=interference,
        )
        
        transmission_rate = compute_transmission_rate(SINR=sinr, bandwidth=self._V2V_bandwidth)
        return transmission_rate
