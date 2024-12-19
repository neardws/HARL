from lyapunov_optimization.virtual_queues.baseQueue import baseQueue

class cc_ressource_queue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        cloud_computing_capability,
    ):
        self._cloud_computing_capability = cloud_computing_capability
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        task_offloaded_at_cloud,
        computation_resource_allocation_actions,
    ):  
        input = 0.0
        for index, task in enumerate(task_offloaded_at_cloud):
            allocated_cycles = self._cloud_computing_capability * \
                computation_resource_allocation_actions["cloud"][index]
            input += allocated_cycles
        return input
    
    def compute_output(
        self, 
    ):
        return self._cloud_computing_capability