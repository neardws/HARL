class baseQueue:
    def __init__(
        self,
        time_slot_num: int,
        name: str,
    ):
        self._time_slot_num = time_slot_num
        self._queue = [0] * self._time_slot_num
        self._queue_name = name
        self._inputs = [0] * self._time_slot_num
        self._outputs = [0] * self._time_slot_num

    def reset(self):
        self._queue = [0] * self._time_slot_num
        self._inputs = [0] * self._time_slot_num
        self._outputs = [0] * self._time_slot_num
    
    def update(
        self,
        input: float,
        output: float,
        time_slot: int,
    ):
        if time_slot > self._time_slot_num or time_slot < 0:
            raise ValueError("The time slot is out of range.")
        self._inputs[time_slot] = input
        self._outputs[time_slot] = output
        if time_slot < self._time_slot_num - 1:
            try:
                self._queue[time_slot + 1] = self.max_0(float(self._queue[time_slot] + input - output))
            except TypeError:
                print("input: ", input)
                print("output: ", output)
                print("time_slot: ", time_slot)
                print("queue: ", self._queue[time_slot])
                print("queue: ", self._queue[time_slot] + input)
                print("queue: ", self._queue[time_slot] + input - output)
                print("queue: ", self.max_0(float(self._queue[time_slot] + input - output)))
                raise
        
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
    
    def get_input_by_time(self, time_slot: int):
        if time_slot > self._time_slot_num or time_slot < 0:
            raise ValueError("The time slot is out of range.")
        return self._inputs[time_slot]
    
    def get_output_by_time(self, time_slot: int):
        if time_slot > self._time_slot_num or time_slot < 0:
            raise ValueError("The time slot is out of range.")
        return self._outputs[time_slot]
    
    def __str__(self):
        return "name: " + self._queue_name + "\nqueue: " + str(self._queue) + "\ninputs: " + str(self._inputs) + "\noutputs: " + str(self._outputs) + "\n"

