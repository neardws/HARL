from typing import List
from objects.vehicle_object import vehicle
from lyapunov_optimization.virtual_queues.baseQueue import baseQueue

class delayQueue(baseQueue):
    def __init__(self, time_slot_num, name, task_index, client_vehicle_number, maximum_task_generation_number):
        self._task_index = task_index
        self._client_vehicle_number = client_vehicle_number
        self._maximum_task_generation_number = maximum_task_generation_number
        super().__init__(time_slot_num, name)
        
    def compute_input(
        self, 
        client_vehicles: List[vehicle],
        now: int,
        task_offloading_decisions,
        lc_queue_backlogs,
        v2v_queue_backlogs,
        vc_queue_backlogs,
        v2i_queue_backlogs,
        i2i_queue_backlogs,
        ec_queue_backlogs,
        i2c_queue_backlogs,
        cc_queue_backlogs,
    ):  
        for client_vehicle_index in range(self._client_vehicle_number):
            task_of_client_vehicle = client_vehicles[client_vehicle_index].get_tasks_by_time(now)
            input = 0.0
            min_num = min(len(task_of_client_vehicle), self._maximum_task_generation_number)
            if min_num > 0:
                for j in range(min_num):
                    if task_of_client_vehicle[j][1] == self._task_index:
                        input += self.compute_total_queue_length(
                            client_vehicle_index,
                            self._task_index,
                            task_offloading_decisions,
                            lc_queue_backlogs,
                            v2v_queue_backlogs,
                            vc_queue_backlogs,
                            v2i_queue_backlogs,
                            i2i_queue_backlogs,
                            ec_queue_backlogs,
                            i2c_queue_backlogs,
                            cc_queue_backlogs,
                        )
        return input
                    
    def compute_total_queue_length(
        self,
        client_vehicle_index: int,
        task_index: int,
        task_offloading_decisions,
        lc_queue_backlogs,
        v2v_queue_backlogs,
        vc_queue_backlogs,
        v2i_queue_backlogs,
        i2i_queue_backlogs,
        ec_queue_backlogs,
        i2c_queue_backlogs,
        cc_queue_backlogs,
    ):
        task_offloading = task_offloading_decisions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(task_index)]
        if task_offloading == "Local":
            return lc_queue_backlogs[client_vehicle_index]
        elif task_offloading.startswith("Server Vehicle "):
            server_vehicle_index = int(task_offloading.split(" ")[-1])
            return v2v_queue_backlogs[server_vehicle_index] + vc_queue_backlogs[server_vehicle_index]
        elif task_offloading.startswith("Edge Node "):
            edge_node_index = int(task_offloading.split(" ")[-1])
            return max(v2i_queue_backlogs[edge_node_index], i2i_queue_backlogs[edge_node_index]) + ec_queue_backlogs[edge_node_index]
        elif task_offloading == "Cloud":
            return i2c_queue_backlogs[client_vehicle_index] + cc_queue_backlogs[client_vehicle_index]
        else:
            raise ValueError("Invalid task offloading decision")
        
    """
    compute the average queue length of the delay queue by
    the average data size of the tasks in the queue and
    the average task arrival rate of the tasks in the queue
    """
    def compute_output(
        self,
        client_vehicles : List[vehicle],
    ): 
        average_task_arrival_rate = 0.0
        average_data_size = 0.0
        
        for client_vehicle in client_vehicles:
            average_task_arrival_rate += client_vehicle.get_task_arrival_rate_by_task_index(self._task_index)
            average_data_size += client_vehicle.get_average_task_data_size_by_task_index(self._task_index)
        
        average_task_arrival_rate /= self._client_vehicle_number
        average_data_size /= self._client_vehicle_number
          
        return average_task_arrival_rate * average_data_size