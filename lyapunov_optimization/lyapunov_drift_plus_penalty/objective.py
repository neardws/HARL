from typing import List
from lyapunov_optimization.virtual_queues.delay_queue import delayQueue
from lyapunov_optimization.virtual_queues.local_computing_resource_queue import lc_ressource_queue
from lyapunov_optimization.virtual_queues.vehicle_computing_resource_queue import vc_ressource_queue
from lyapunov_optimization.virtual_queues.edge_computing_resource_queue import ec_ressource_queue
from lyapunov_optimization.virtual_queues.cloud_compjuting_resource_queue import cc_ressource_queue

def compute_phi_t(
    now: int,
    task_number: int,
    delay_queues: List[delayQueue],
    client_vehicle_number: int,
    lc_queues: List[lc_ressource_queue],
    server_vehicle_number: int,
    vc_queues: List[vc_ressource_queue],
    edge_node_number: int,
    ec_queues: List[ec_ressource_queue],
    cc_queue: cc_ressource_queue,
):
    phi_t = 0.0
    
    for i in range(task_number):
        phi_t += (delay_queues[i].get_queue(now) - delay_queues[i].get_output_by_time(now)) * \
            delay_queues[i].get_input_by_time(now)
    for i in range(client_vehicle_number):
        phi_t += (lc_queues[i].get_queue(now) - lc_queues[i].get_output_by_time(now)) * \
            lc_queues[i].get_input_by_time(now)
    for i in range(server_vehicle_number):
        phi_t += (vc_queues[i].get_queue(now) - vc_queues[i].get_output_by_time(now)) * \
            vc_queues[i].get_input_by_time(now)
    for i in range(edge_node_number):
        phi_t += (ec_queues[i].get_queue(now) - ec_queues[i].get_output_by_time(now)) * \
            ec_queues[i].get_input_by_time(now)
    phi_t += (cc_queue.get_queue(now) - cc_queue.get_output_by_time(now)) * cc_queue.get_input_by_time(now)
    
    return phi_t