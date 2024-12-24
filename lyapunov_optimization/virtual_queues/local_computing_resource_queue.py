from lyapunov_optimization.virtual_queues.baseQueue import baseQueue
from utilities.conversion import cover_GHz_to_Hz

class lc_ressource_queue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        client_vehicle_index,
        client_vehicle_computing_capability,
    ):
        self._client_vehicle_index = client_vehicle_index
        self._client_vehicle_computing_capability = client_vehicle_computing_capability
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        task_offloaded_at_client_vehicles,
        computation_resource_allocation_actions,
    ):  
        input = 0.0
        for index, task in enumerate(task_offloaded_at_client_vehicles["client_vehicle_" + str(self._client_vehicle_index)]):
            allocated_cycles = cover_GHz_to_Hz(self._client_vehicle_computing_capability) * \
                computation_resource_allocation_actions["client_vehicle_" + str(self._client_vehicle_index)][index]
            input += allocated_cycles 
        return input
    
    def compute_output(
        self, 
    ):
        return cover_GHz_to_Hz(self._client_vehicle_computing_capability)


