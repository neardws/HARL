from typing import List
from objects.task_object import task
from queues.base_queue import baseQueue

class ECQueue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        edge_node_index,
    ):
        self._edge_node_index = edge_node_index
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        v2i_transmission_output,
        i2i_transmission_output, 
    ):
        input = v2i_transmission_output + i2i_transmission_output
        return input
    
    def compute_output(
        self, 
        tasks : List[task],
        edge_node_computing_capability,
        computation_resource_allocation_actions,
        task_offloaded_at_edge_nodes,
    ):
        output = 0.0
        for task, index in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            task_index = task["task_index"]
            task_required_cycles = tasks[task_index].get_requested_computing_cycles()
            allocated_cycles = edge_node_computing_capability * \
                computation_resource_allocation_actions["edge_node_" + str(self._edge_node_index)][index]
            output += allocated_cycles / task_required_cycles
        return output
    
    
    
    