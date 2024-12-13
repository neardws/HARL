from typing import List
from objects.task_object import task
from objects.vehicle_object import vehicle
from queues.base_queue import baseQueue

class VCQueue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        server_vehicle_index,
    ):
        self._server_vehicle_index = server_vehicle_index
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        local_computing_input,
        v2v_transmission_output, 
    ):
        input = local_computing_input + v2v_transmission_output
        return input
    
    def compute_output(
        self, 
        tasks : List[task],
        server_vehicles : List[vehicle],
        computation_resource_allocation_actions,
        task_offloaded_at_server_vehicles,
    ):
        output = 0.0
        for task, index in enumerate(task_offloaded_at_server_vehicles["server_vehicle_" + str(self._server_vehicle_index)]):
            task_index = task["task_index"]
            task_required_cycles = tasks[task_index].get_requested_computing_cycles()
            allocated_cycles = server_vehicles[self._server_vehicle_index].get_computing_capability() * \
                computation_resource_allocation_actions["server_vehicle_" + str(self._server_vehicle_index)][index]
            output += allocated_cycles / task_required_cycles
        return output
    
    
    
    