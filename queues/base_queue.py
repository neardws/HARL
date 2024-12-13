class baseQueue:
    def __init__(
        self,
        time_slot_num: int,
        name: str,
    ):
        self._time_slot_num = time_slot_num
        self._queue = [0] * self._time_slot_num
        self._queue_name = name

    def update(
        self,
        input: float,
        output: float,
        time_slot: InterruptedError,
    ):
        if time_slot > self._time_slot_num or time_slot < 0:
            raise ValueError("The time slot is out of range.")
        self._queue[time_slot + 1] = self.max_0(self._queue[time_slot] - output) + input
        
    def max_0(self, x):
        return max(0, x)
    
    def compute_input(self):
        raise NotImplementedError
    
    def compute_output(self):
        raise NotImplementedError

    def get_queue(self, time_slot: int):
        if time_slot > self._time_slot_num or time_slot < 0:
            raise ValueError("The time slot is out of range.")
        return self._queue[time_slot]
