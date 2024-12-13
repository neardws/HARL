from typing import List, Tuple
import random
import numpy as np
from objects.mobility_object import mobility
from typing import Optional

class vehicle(object):
    '''
    the vehicle is defined by its mobilities, computing capability, available computing capability and transmission power
    args:
        mobilities: a list of mobility
        computing_capability: computing capability
        available_computing_capability: available computing capability
        transmission_power: transmission power
    '''
    # TODO: update the generation of vehicles in utilities/object_generation.py
    def __init__(
        self,
        random_seed: int,
        mobilities: List[mobility],
        computing_capability: float,
        storage_capability: float,
        time_slot_num: int,
        transmission_power: float,
        communication_range: float,
        min_task_arrival_rate: float,
        max_task_arrival_rate: float,
        task_num: int,
        task_ids_rate: float,
    ) -> None:
        self._mobilities : List[mobility] = mobilities
        self._computing_capability : float = computing_capability
        self._storage_capability : float = storage_capability
        self._time_slot_num : int = time_slot_num
        self._available_computing_capability : List[float] = [computing_capability for _ in range(time_slot_num)]
        self._available_storage_capability : List[float] = [storage_capability for _ in range(time_slot_num)]
        self._transmission_power : float = transmission_power
        self._communication_range : float = communication_range
        self._min_task_arrival_rate : float = min_task_arrival_rate
        self._max_task_arrival_rate : float = max_task_arrival_rate
        self._task_num : int = task_num
        self._random_seed : int = random_seed
        random.seed(self._random_seed)
        self._tasks : List[Tuple] = self.generate_task()
        self._task_ids_rate = task_ids_rate
        self._task_ids = self._generate_task_ids(rate=self._task_ids_rate)
        self._task_ids_num = len(self._task_ids)
        self._task_arrival_rate = self.generate_task_arrival_rates()
    
    def get_time_slot_num(self) -> int:
        return self._time_slot_num
    
    def get_mobilities(self) -> List[mobility]:
        return self._mobilities
    
    def get_mobility(self, now: int) -> mobility:
        if now >= self._time_slot_num:
            raise ValueError("The time is out of range.")
        return self._mobilities[now]
    
    def get_computing_capability(self) -> float:
        return self._computing_capability
    
    def get_storage_capability(self) -> float:
        return self._storage_capability
    
    def get_available_computing_capability(self, now: int) -> float:
        return self._available_computing_capability[now]
    
    def get_available_storage_capability(self, now: int) -> float:
        return self._available_storage_capability[now]
    
    def get_transmission_power(self) -> float:
        return self._transmission_power
    
    def get_communication_range(self) -> float:
        return self._communication_range
    
    def get_tasks(self) -> List[Tuple]:
        return self._tasks
    
    def get_tasks_by_time(self, now : int) -> List[Tuple]:
        return [task for task in self._tasks if task[0] == now]
    
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
    
    def generate_task_ids(self, rate=0.1) -> List[int]:
        return random.choices(range(self._task_num), k=int(self._task_num * rate))
    
    def generate_task_arrival_rates(self) -> List[float]:
        # 生成任务到达率, 对每一个task_id生成一个到达率, 根据最大和最小到达率
        return [random.uniform(self._min_task_arrival_rate, self._max_task_arrival_rate) for _ in range(self._task_ids_num)]
    
    def get_task_arrival_rate_by_task_id(self, task_id: int) -> float:
        task_id_index = self._task_ids.index(task_id)
        return self._task_arrival_rate[task_id_index]
    
    def generate_task(self) -> List[Tuple]:
        # 根据不同task_id的到达率生成任务, 需要服从泊松分布 
        tasks = [] 
        for i in range(self._task_ids_num):
            task_arrival_times = np.random.poisson(self._task_arrival_rate[i], self._time_slot_num)
            for j in range(self._time_slot_num):
                if task_arrival_times[j] > 0:
                    for _ in range(task_arrival_times[j]):
                        tasks.append((j, self._task_ids[i]))
        return tasks
    
    def __str__(
        self, 
        now: Optional[int] = 0,
    ) -> str:
        return "location_x: " + str(self.get_mobility(now).get_x()) \
            + "\nlocation_y: " + str(self.get_mobility(now).get_y()) \
            + "\nspeed: " + str(self.get_mobility(now).get_speed()) \
            + "\ndirection: " + str(self.get_mobility(now).get_direction()) \
            + "\ncomputing_capability: " + str(self.get_computing_capability()) \
            + "\nstorage_capability: " + str(self.get_storage_capability()) \
            + "\navailable_computing_capability: " + str(self.get_available_computing_capability(now)) \
            + "\navailable_storage_capability: " + str(self.get_available_storage_capability(now)) \
            + "\ntransmission_power: " + str(self.get_transmission_power()) \
            + "\ncommunication_range: " + str(self.get_communication_range()) \
            + "\ntask_arrival_rate: " + str(self.get_task_arrival_rate())
