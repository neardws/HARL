from queues.base_queue import baseQueue
from utilities.conversion import cover_GHz_to_Hz, cover_MB_to_bit

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
        cloud_computing_capability,
        computation_resource_allocation_actions,
        task_offloaded_at_cloud,
    ):
        output = 0.0
        for index, task in enumerate(task_offloaded_at_cloud["cloud"]):
            task_required_cycles = task["task"].get_requested_computing_cycles()
            task_size = cover_MB_to_bit(task["task"].get_input_data_size())
            allocated_cycles = cover_GHz_to_Hz(cloud_computing_capability * \
                computation_resource_allocation_actions["cloud"][index])
            output += allocated_cycles / task_required_cycles * task_size  
        return output