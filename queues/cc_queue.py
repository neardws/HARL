from typing import List
from objects.task_object import task
from queues.base_queue import baseQueue

class CCQueue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
    ):
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        i2c_transmission_output,
    ):
        input = i2c_transmission_output
        return input
    
    def compute_output(
        self, 
        tasks : List[task],
        cloud_computing_capability,
        computation_resource_allocation_actions,
        task_offloaded_at_edge_nodes,
    ):
        output = 0.0
        for task, index in enumerate(task_offloaded_at_edge_nodes["cloud"]):
            task_index = task["task_index"]
            task_required_cycles = tasks[task_index].get_requested_computing_cycles()
            allocated_cycles = cloud_computing_capability * \
                computation_resource_allocation_actions["cloud"][index]
            output += allocated_cycles / task_required_cycles
        return output