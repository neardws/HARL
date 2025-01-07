import numpy as np
from typing import List, Dict, Tuple
from itertools import product

class DPAgent:
    def __init__(
        self,
        client_vehicle_num: int,
        maximum_task_generation_number_of_vehicles: int,
        maximum_server_vehicle_num: int,
        edge_num: int,
        max_action: int,
        agent_number: int,
        state_horizon: int = 3,  # 考虑的未来步数
        discount_factor: float = 0.9,
        threshold: float = 1e-4,  # 收敛阈值
    ):
        self._client_vehicle_num = client_vehicle_num
        self._maximum_task_generation_number_of_vehicles = maximum_task_generation_number_of_vehicles
        self._maximum_server_vehicle_num = maximum_server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + maximum_server_vehicle_num + edge_num + 1  # 本地、车辆、边缘、云
        self._max_action = max_action
        self._agent_number = agent_number
        self._state_horizon = state_horizon
        self._discount_factor = discount_factor
        self._threshold = threshold

        # 初始化策略和值函数
        self.policy = {}
        self.value_function = {}

        # 构建状态空间和动作空间
        self.state_space = self._build_state_space()
        self.action_space = self._build_action_space()

        # 进行值迭代以优化策略
        self._value_iteration()

    def generate_action(self, obs: List[np.ndarray]) -> np.ndarray:
        """
        根据当前观测生成动作，使用预先计算的最优策略。

        Args:
            obs: 每个智能体的观测值列表。

        Returns:
            actions: 每个智能体的动作数组。
        """
        actions = np.zeros((self._agent_number, self._max_action))

        # 任务卸载智能体 (agent_index = 0)
        task_offloading_actions = self._get_optimal_offloading_actions(obs[0])
        actions[0] = task_offloading_actions

        # 资源分配智能体 (agent_index = 1)
        resource_allocation_actions = self._get_optimal_resource_allocation_actions(obs[1])
        actions[1] = resource_allocation_actions

        return actions

    def _build_state_space(self) -> List[Tuple]:
        """
        构建所有可能的状态空间。

        Returns:
            state_space: 状态空间的列表。
        """
        # 简化状态空间：每个队列的积压量取0或1
        # 实际应用中应根据具体情况扩展状态表示
        lc_states = [0, 1]  # 本地计算队列
        v2v_states = [0, 1] * self._maximum_server_vehicle_num  # V2V连接状态
        v2i_states = [0, 1] * self._edge_num  # V2I连接状态
        ec_states = [0, 1] * self._edge_num  # 边缘计算队列

        state_space = list(product(lc_states, 
                                   *[v2v_states[i:i+self._maximum_server_vehicle_num] for i in range(0, len(v2v_states), self._maximum_server_vehicle_num)],
                                   *[v2i_states[i:i+self._edge_num] for i in range(0, len(v2i_states), self._edge_num)],
                                   *[ec_states[i:i+self._edge_num] for i in range(0, len(ec_states), self._edge_num)]))
        return state_space

    def _build_action_space(self) -> List[Tuple]:
        """
        构建所有可能的动作空间。

        Returns:
            action_space: 动作空间的列表。
        """
        # 简化动作空间：每个任务的卸载选择（0: 本地, 1: 服务器, 2: 边缘, 3: 云）
        # 资源分配动作为离散选择（例如，分配给不同资源模块）
        task_offloading_actions = list(range(self._task_offloading_number))
        resource_allocation_actions = list(range(2))  # 示例：0 或 1

        # 组合所有智能体的动作
        action_space = list(product(task_offloading_actions, resource_allocation_actions))
        return action_space

    def _value_iteration(self):
        """
        使用值迭代算法优化策略。
        """
        # 初始化值函数为0
        self.value_function = {state: 0 for state in self.state_space}
        self.policy = {state: 0 for state in self.state_space}

        while True:
            delta = 0
            new_value_function = self.value_function.copy()

            for state in self.state_space:
                # 对每个可能的动作计算期望值
                action_values = []
                for action in self.action_space:
                    next_state, reward = self._transition(state, action)
                    action_value = reward + self._discount_factor * self.value_function[next_state]
                    action_values.append(action_value)

                # 选择最大值作为新的值函数
                max_action_value = max(action_values)
                best_action = np.argmax(action_values)

                # 更新值函数和策略
                new_value_function[state] = max_action_value
                self.policy[state] = best_action

                # 更新最大变化量
                delta = max(delta, abs(self.value_function[state] - new_value_function[state]))

            self.value_function = new_value_function

            # 检查是否收敛
            if delta < self._threshold:
                print("值迭代收敛, 停止迭代。")
                break

    def _transition(self, state: Tuple, action: Tuple) -> Tuple[Tuple, float]:
        """
        定义环境的状态转移和奖励。

        Args:
            state: 当前状态。
            action: 当前动作。

        Returns:
            next_state: 转移后的下一个状态。
            reward: 当前动作的奖励。
        """
        # 简化的状态转移逻辑
        # 实际应用中需要根据具体环境定义
        # 这里假设动作影响队列的积压情况

        # 解码动作
        task_offloading_action, resource_allocation_action = action

        # 定义奖励：负的队列积压总和，队列越少奖励越高
        reward = -sum(state)

        # 定义下一状态：根据动作调整队列
        # 这是一个示例，实际逻辑需根据具体环境设计
        next_state = list(state)
        # 示例逻辑：如果选择本地处理，则减少本地队列
        if task_offloading_action == 0 and state[0] > 0:
            next_state = list(next_state)
            next_state[0] -= 1
        # 如果选择服务器，则减少相应的V2V队列
        elif 1 <= task_offloading_action <= self._maximum_server_vehicle_num:
            server_idx = task_offloading_action - 1
            if state[1 + server_idx] > 0:
                next_state = list(next_state)
                next_state[1 + server_idx] -= 1
        # 如果选择边缘，则减少相应的V2I队列
        elif 1 + self._maximum_server_vehicle_num <= task_offloading_action < 1 + self._maximum_server_vehicle_num + self._edge_num:
            edge_idx = task_offloading_action - 1 - self._maximum_server_vehicle_num
            if state[1 + self._maximum_server_vehicle_num + edge_idx] > 0:
                next_state = list(next_state)
                next_state[1 + self._maximum_server_vehicle_num + edge_idx] -= 1
        # 如果选择云，则减少云队列
        elif task_offloading_action == self._task_offloading_number - 1:
            if state[-1] > 0:
                next_state = list(next_state)
                next_state[-1] -= 1

        # 随机增减队列
        # 这里假设有一定概率任务生成，增加队列
        prob_new_task = 0.3
        for i in range(len(next_state)):
            if np.random.rand() < prob_new_task:
                next_state[i] = min(next_state[i] + 1, 1)  # 限制队列积压最大为1

        return tuple(next_state), reward

    def _get_optimal_offloading_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        根据最优策略生成任务卸载动作。

        Args:
            obs: 任务卸载智能体的观测值。

        Returns:
            task_offloading_actions: 归一化的任务卸载动作。
        """
        # 根据当前观测确定当前状态
        state = self._map_obs_to_state(obs)

        # 获取策略中的最佳动作
        best_action_index = self.policy.get(state, 0)
        best_action = self.action_space[best_action_index]

        # 解析动作
        task_offloading_action, _ = best_action

        # 将动作转换为归一化值
        normalized_action = task_offloading_action / (self._task_offloading_number - 1)
        return normalized_action

    def _get_optimal_resource_allocation_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        根据最优策略生成资源分配动作。

        Args:
            obs: 资源分配智能体的观测值。

        Returns:
            resource_allocation_actions: 归一化的资源分配动作。
        """
        # 根据当前观测确定当前状态
        state = self._map_obs_to_state(obs)

        # 获取策略中的最佳动作
        best_action_index = self.policy.get(state, 0)
        best_action = self.action_space[best_action_index]

        # 解析动作
        _, resource_allocation_action = best_action

        # 将动作转换为归一化值
        normalized_action = resource_allocation_action / (len(self._action_space) - 1)
        return normalized_action

    def _map_obs_to_state(self, obs: np.ndarray) -> Tuple:
        """
        将观测值映射到状态。

        Args:
            obs: 观测值数组。

        Returns:
            state: 对应的状态元组。
        """
        # 简化的映射逻辑，实际应用中需根据具体观测定义
        lc_queue = 1 if obs[0] > 0.5 else 0
        v2v_conn = tuple(1 if x > 0.5 else 0 for x in obs[1:self._maximum_server_vehicle_num + 1])
        v2i_conn = tuple(1 if x > 0.5 else 0 for x in obs[self._maximum_server_vehicle_num + 1:self._maximum_server_vehicle_num + 1 + self._edge_num])
        ec_queue = tuple(1 if x > 0.5 else 0 for x in obs[self._maximum_server_vehicle_num + 1 + self._edge_num:self._maximum_server_vehicle_num + 1 + 2 * self._edge_num])
        cloud_queue = 1 if obs[-1] > 0.5 else 0

        state = (lc_queue,) + v2v_conn + v2i_conn + ec_queue + (cloud_queue,)
        return state

    def _get_all_possible_actions(self) -> List[Tuple]:
        """
        获取所有可能的动作组合。

        Returns:
            actions: 动作的列表。
        """
        return self.action_space

    def _compute_reward(self, state: Tuple, action: Tuple, next_state: Tuple) -> float:
        """
        计算从当前状态执行动作后获得的奖励。

        Args:
            state: 当前状态。
            action: 当前动作。
            next_state: 执行动作后的下一个状态。

        Returns:
            reward: 奖励值。
        """
        # 奖励定义为负的总队列积压，队列越少奖励越高
        reward = -sum(next_state)
        return reward

    def _compute_transition_probability(self, state: Tuple, action: Tuple, next_state: Tuple) -> float:
        """
        计算从当前状态执行动作后转移到下一个状态的概率。

        Args:
            state: 当前状态。
            action: 当前动作。
            next_state: 下一个状态。

        Returns:
            prob: 转移概率。
        """
        # 简化的转移概率模型
        # 如果动作导致状态变化，则概率为1，否则为0
        reward = self._compute_reward(state, action, next_state)
        if self._transition(state, action)[0] == next_state:
            return 1.0
        else:
            return 0.0

    def _compute_all_possible_transitions(self, state: Tuple, action: Tuple) -> List[Tuple[Tuple, float]]:
        """
        获取给定状态和动作的所有可能转移及其概率。

        Args:
            state: 当前状态。
            action: 当前动作。

        Returns:
            transitions: 列表，包含所有可能的(下一个状态, 概率)。
        """
        # 在简化模型中，每个动作导致确定的下一个状态
        next_state, reward = self._transition(state, action)
        transitions = [(next_state, 1.0)]
        return transitions

    def _map_policy_to_actions(self, state: Tuple) -> Tuple:
        """
        根据策略映射状态到动作。

        Args:
            state: 当前状态。

        Returns:
            action: 对应的动作。
        """
        action_index = self.policy.get(state, 0)
        action = self.action_space[action_index]
        return action 