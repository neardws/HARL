from typing import List, Tuple
from objects.vehicle_object import vehicle

def get_client_and_server_vehicles(
    vehicles: List[vehicle],
) -> Tuple[List[vehicle], List[vehicle]]:
    client_vehicles = []
    server_vehicles = []
    for vehicle in vehicles:
        if vehicle._task_ids_num == 0:
            server_vehicles.append(vehicle)
        else:
            client_vehicles.append(vehicle)
    return client_vehicles, server_vehicles