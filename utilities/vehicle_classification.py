from typing import List, Tuple
from objects.vehicle_object import vehicle

def get_client_and_server_vehicles(
    now: int,
    vehicles: List[vehicle],
) -> Tuple[List[vehicle], List[vehicle]]:
    client_vehicles = []
    server_vehicles = []
    for vehicle in vehicles:
        if vehicle.get_tasks_by_time(now) != []:
            client_vehicles.append(vehicle)
        else:
            server_vehicles.append(vehicle)
    return client_vehicles, server_vehicles