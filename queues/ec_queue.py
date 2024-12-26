from queues.base_queue import baseQueue
from utilities.conversion import cover_GHz_to_Hz, cover_MB_to_bit

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
        v2i_transmission_input,
        v2i_transmission_output,
        i2i_transmission_input,
        i2i_transmission_output, 
    ):
        input = min(v2i_transmission_input, v2i_transmission_output) + \
            min(i2i_transmission_input, i2i_transmission_output)
        return input
    
    def compute_output(
        self, 
        edge_node_computing_capability,
        computation_resource_allocation_actions,
        task_offloaded_at_edge_nodes,
    ):
        output = 0.0
        for index, task in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            task_required_cycles = task["task"].get_requested_computing_cycles()
            task_size = cover_MB_to_bit(task["task"].get_input_data_size())
            allocated_cycles =cover_GHz_to_Hz(edge_node_computing_capability * \
                computation_resource_allocation_actions["edge_node_" + str(self._edge_node_index)][index])
            output += allocated_cycles / task_required_cycles * task_size
        return output
    
    
    
    