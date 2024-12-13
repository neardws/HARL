from typing import List
from objects.mobility_object import mobility
class edge_node(object):
    
    def __init__(
        self,
        edge_node_mobility: mobility,       
        computing_capability: float,
        storage_capability: float,
        communication_range: float,
        time_slot_num: int,
    ) -> None:
        self._mobility: mobility = edge_node_mobility 
        self._computing_capability: float = computing_capability
        self._time_slot_num: int = time_slot_num
        self._available_computing_capability: List[float] = [self._computing_capability for _ in range(self._time_slot_num)] 
        self._storage_capability : float = storage_capability
        self._available_storage_capability: List[float] = [self._storage_capability for _ in range(self._time_slot_num)]
        self._communication_range : float = communication_range

    def get_mobility(self) -> mobility:
        return self._mobility
    
    def get_computing_capability(self) -> float:
        return self._computing_capability
    
    def get_storage_capability(self) -> float:
        return self._storage_capability
    
    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_available_computing_capability(self, now: int) -> float:
        return self._available_computing_capability[now]
    
    def get_available_storage_capability(self, now: int) -> float:
        return self._available_storage_capability[now]

    def set_consumed_computing_capability(self, consumed_computing_capability: float, now: int, duration: int) -> None:
        if now > self._time_slot_num:
            return None
        if now + duration > self._time_slot_num:
            end_time = self._time_slot_num
        else:
            end_time = now + duration
        for i in range(int(now), int(end_time)):
            self._available_computing_capability[i] = self._available_computing_capability[i] - consumed_computing_capability
        return None    

    def set_consumed_storage_capability(self, consumed_storage_capability: float, now: int, duration: int) -> None:
        if now > self._time_slot_num:
            return None
        if now + duration > self._time_slot_num:
            end_time = self._time_slot_num
        else:
            end_time = now + duration
        for i in range(int(now), int(end_time)):
            self._available_storage_capability[i] = self._available_storage_capability[i] - consumed_storage_capability
        return None
        
