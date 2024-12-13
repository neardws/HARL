from queues.base_queue import baseQueue

class LCQueue(baseQueue):
    def __init__(
        self, 
        time_slot_num, 
        name,
        client_vehicle_index,
        client_vehicles,
        maximum_task_generation_number,
        maximum_task_offloaded_at_client_vehicle_number,
        tasks,
    ):
        self._client_vehicle_index = client_vehicle_index
        self._client_vehicles = client_vehicles
        self._maximum_task_generation_number = maximum_task_generation_number
        self._maximum_task_offloaded_at_client_vehicle_number = maximum_task_offloaded_at_client_vehicle_number
        self._tasks = tasks
        super().__init__(time_slot_num, name)
    
    def compute_input(
        self, 
        task_offloading_actions, 
        now
    ):
        input = 0.0
        tasks_of_my = self._client_vehicles[self._client_vehicle_index].get_tasks_by_time(now)
        min_num = min(len(tasks_of_my), self._maximum_task_generation_number)
        if min_num > 0:
            for j in range(min_num):
                if task_offloading_actions["client_vehicle_" + str(self._client_vehicle_index) + "_task_" + str(j)] == "Local":
                    task_id = tasks_of_my[j][1]
                    task_size = self._tasks[task_id].get_input_data_size()
                    task_arrival_rate = self._client_vehicles[self._client_vehicle_index].get_task_arrival_rate_by_task_id(task_id)
                    input += task_size * task_arrival_rate
        return input
    
    def compute_output(
        self, 
        task_offloading_actions, 
        computation_resource_allocation_actions,
    ):
        output = 0.0
        tasks_of_my = self._client_vehicles[self._client_vehicle_index].get_tasks_by_time(now)
        min_num = min(len(tasks_of_my), self._maximum_task_generation_number)
        task_offloaded_number = 0
        if min_num > 0:
            for j in range(min_num):
                if task_offloading_actions["client_vehicle_" + str(self._client_vehicle_index) + "_task_" + str(j)] == "Local":
                    task_id = tasks_of_my[j][1]
                    task_required_cycles = self._tasks[task_id].get_requested_computing_cycles()
                    allocated_cycles = self._client_vehicles[self._client_vehicle_index].get_computing_capability() * \
                        computation_resource_allocation_actions["client_vehicle_" + str(self._client_vehicle_index)][task_offloaded_number]
                    output += allocated_cycles / task_required_cycles
                    task_offloaded_number += 1
                    if task_offloaded_number > self._maximum_task_offloaded_at_client_vehicle_number:
                        break
        return output
    
    
    
    