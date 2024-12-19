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
        v2v_transmission_output, 
    ):
        input = v2v_transmission_output
        return input
    
    def compute_output(
        self, 
        server_vehicle_compute_capability,
        computation_resource_allocation_actions,
        task_offloaded_at_server_vehicles,
    ):
        output = 0.0
        for index, task in enumerate(task_offloaded_at_server_vehicles["server_vehicle_" + str(self._server_vehicle_index)]):
            task_required_cycles = task["task"].get_requested_computing_cycles()
            allocated_cycles = server_vehicle_compute_capability * \
                computation_resource_allocation_actions["server_vehicle_" + str(self._server_vehicle_index)][index]
            output += allocated_cycles / task_required_cycles
        return output