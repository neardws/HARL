from lyapunov_optimization.virtual_queues.baseQueue import baseQueue

class ec_ressource_queue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        edge_node_index,
        edge_node_computing_capability,
    ):
        self._edge_node_index = edge_node_index
        self._edge_node_computing_capability = edge_node_computing_capability
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        task_offloaded_at_edge_nodes,
        computation_resource_allocation_actions,
    ):  
        input = 0.0
        for index, task in enumerate(task_offloaded_at_edge_nodes["edge_node_" + str(self._edge_node_index)]):
            allocated_cycles = self._edge_node_computing_capability * \
                computation_resource_allocation_actions["edge_node_" + str(self._edge_node_index)][index]
            input += allocated_cycles
        return input
    
    def compute_output(
        self, 
    ):
        return self._edge_node_computing_capability