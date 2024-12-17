from typing import List
from objects.task_object import task
from objects.vehicle_object import vehicle
from queues.base_queue import baseQueue

class LCQueue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        client_vehicle_index,
        maximum_task_generation_number,
    ):
        self._client_vehicle_index = client_vehicle_index
        self._maximum_task_generation_number = maximum_task_generation_number
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        client_vehicle : vehicle,
        task_offloaded_at_client_vehicles,
    ):
        input = 0.0
        for task, index in enumerate(task_offloaded_at_client_vehicles["client_vehicle_" + str(self._client_vehicle_index)]):
            task_index = task["task_index"]
            task_size = task["task"].get_input_data_size()
            task_arrival_rate = client_vehicle.get_task_arrival_rate_by_task_index(task_index)
            input += task_size * task_arrival_rate
        return input
    
    def compute_output(
        self, 
        computation_resource_allocation_actions,
        task_offloaded_at_client_vehicles,
        client_vehicle_computing_capability, # self._client_vehicles[self._client_vehicle_index].get_computing_capability()
    ):
        output = 0.0
        for task, index in enumerate(task_offloaded_at_client_vehicles["client_vehicle_" + str(self._client_vehicle_index)]):
            task_required_cycles = task["task"].get_requested_computing_cycles()
            allocated_cycles = client_vehicle_computing_capability * \
                computation_resource_allocation_actions["client_vehicle_" + str(self._client_vehicle_index)][index]
            output += allocated_cycles / task_required_cycles
        return output
    
    
    
    