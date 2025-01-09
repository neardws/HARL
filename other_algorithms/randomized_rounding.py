import numpy as np

class randomized_rounding_agent(object):

    def __init__(
        self, 
        client_vehicle_num: int,
        maximum_task_generation_number_of_vehicles: int,
        maximum_server_vehicle_num: int,
        edge_num: int,
        max_action: int,
        agent_number: int
    ):
        self._client_vehicle_num = client_vehicle_num
        self._maximum_task_generation_number_of_vehicles = maximum_task_generation_number_of_vehicles
        self._maximum_server_vehicle_num = maximum_server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + self._maximum_server_vehicle_num + self._edge_num + 1
        self._max_action = max_action
        self._agent_number = agent_number

    def generate_action(self):
        actions = np.zeros((self._agent_number, self._max_action))
        for _ in range(self._agent_number):
            if _ == 0:              # task offloading agent
                for i in range(self._client_vehicle_num):
                    for j in range(self._maximum_task_generation_number_of_vehicles):
                        continuous_action = np.random.rand(self._task_offloading_number)
                        binary_action = self.randomized_rounding_single_one(continuous_action)
                        # 将二进制动作转换为任务卸载索引
                        task_offloading_index = np.argmax(binary_action)
                        # 将任务卸载索引转换为任务卸载动作 [0, 1]
                        task_offloading_action = task_offloading_index / (self._task_offloading_number - 1)
                        actions[_, i * self._maximum_task_generation_number_of_vehicles + j] = task_offloading_action 
            elif _ == 1:                 # resouce allocation agent
                action = np.random.rand(self._max_action)
                actions[_, :] = action
            else:
                raise ValueError("Agent number is out of range")
        return actions

    def randomized_rounding_single_one(self, continuous_action):
        # 归一化概率向量
        prob_sum = np.sum(continuous_action)
        if prob_sum == 0:
            # 如果所有概率为0，随机选择一个索引设置为1
            selected_index = np.random.randint(0, len(continuous_action))
            binary_action = np.zeros_like(continuous_action)
            binary_action[selected_index] = 1
        else:
            prob = continuous_action / prob_sum
            # 使用多项分布采样
            binary_action = np.random.multinomial(1, prob)
            # 或者使用以下方式：
            # selected_index = np.random.choice(len(continuous_action), p=prob)
            # binary_action = np.zeros_like(continuous_action)
            # binary_action[selected_index] = 1
        
        return binary_action