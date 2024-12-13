from typing import List

class cloud_server(object):
    def __init__(
        self,
        computing_capability: float,
        storage_capability: float,
        time_slot_num: int,
        wired_bandwidths: List[float],
        ) -> None:
        self._computing_capability : float = computing_capability
        self._storage_capability : float = storage_capability
        self._time_slot_num : int = time_slot_num
        self._available_computing_capability : List[float] = [computing_capability for _ in range(self._time_slot_num)]
        self._available_storage_capability : List[float] = [storage_capability for _ in range(self._time_slot_num)]   
        self._wired_bandwidth : List[float] = wired_bandwidths
        
    def get_computing_capability(self) -> float:
        return self._computing_capability
    
    def get_storage_capability(self) -> float:
        return self._storage_capability
    
    def get_available_computing_capability(self, now: int) -> float:
        return self._available_computing_capability[now]
    
    def get_available_storage_capability(self, now: int) -> float:
        return self._available_storage_capability[now]
    
    def get_wired_bandwidth_between_edge_node_and_cloud(self, edge_node_index: int) -> float:
        return self._wired_bandwidth[edge_node_index]
    
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
    
