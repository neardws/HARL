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
        state_horizon: int = 3,
        discount_factor: float = 0.9,
        threshold: float = 1e-4,
        use_dp: bool = False,  # 是否使用动态规划
        warmup_steps: int = 1000,  # 预热步数
    ):
        self._client_vehicle_num = client_vehicle_num
        self._maximum_task_generation_number_of_vehicles = maximum_task_generation_number_of_vehicles
        self._maximum_server_vehicle_num = maximum_server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + maximum_server_vehicle_num + edge_num + 1
        self._max_action = max_action
        self._agent_number = agent_number
        self._state_horizon = state_horizon
        self._discount_factor = discount_factor
        self._threshold = threshold
        self._use_dp = use_dp
        self._warmup_steps = warmup_steps
        self._step_count = 0
        
        # 存储状态转移样本
        self.state_transitions = {}  # {(state, action): [(next_state, reward), ...]}

        # 初始化策略和值函数
        self.policy = {}
        self.value_function = {}
        
        # 构建动作空间
        self.action_space = self._build_action_space()
        
        # 移除自动调用值迭代
        # self._value_iteration()  # 删除这行

    def generate_action(self, obs: List[np.ndarray]) -> np.ndarray:
        """生成动作，并在预热后进行值迭代"""
        self._step_count += 1
        
        # 生成动作
        if not self._use_dp or self._step_count < self._warmup_steps:
            # 使用启发式策略
            actions = self._generate_heuristic_actions(obs)
        else:
            if self._step_count >= self._warmup_steps:
                print("Starting value iteration...")
                self._value_iteration()
            # 使用优化后的策略
            actions = self._generate_optimal_actions(obs)
        
        return actions
    
    def store_transition(self, obs: List[np.ndarray], current_state: tuple, actions: np.ndarray, env_info: dict = None):
        # 如果有环境信息，存储转移样本
        if env_info is not None and self._step_count < self._warmup_steps:
            reward = env_info.get('reward', 0)
            next_obs = env_info.get('next_obs', obs)
            next_state = self._get_state_key(next_obs[0])
            action_key = tuple(actions.flatten())
            
            if (current_state, action_key) not in self.state_transitions:
                self.state_transitions[(current_state, action_key)] = []
            self.state_transitions[(current_state, action_key)].append((next_state, reward))

    def _generate_heuristic_actions(self, obs: List[np.ndarray]) -> np.ndarray:
        """使用随机策略生成动作"""
        actions = np.zeros((self._agent_number, self._max_action))
        for i in range(self._agent_number):
            actions[i] = np.random.rand(self._max_action)
        return actions

    def _generate_optimal_actions(self, obs: List[np.ndarray]) -> np.ndarray:
        """使用优化后的策略生成动作"""
        actions = np.zeros((self._agent_number, self._max_action))
        state_key = self._get_state_key(obs[0])
        
        if state_key in self.policy:
            optimal_action = self.policy[state_key]
            # 转换为连续动作
            actions[0][0] = optimal_action[0] / (self._task_offloading_number - 1)
            actions[1][0] = optimal_action[1]
        else:
            # 对于未见过的状态使用启发式策略
            return self._generate_heuristic_actions(obs)
        
        return actions

    def _get_state_key(self, obs: np.ndarray) -> tuple:
        """将观测值离散化为状态键
        
        Args:
            obs: 观测值数组
            
        Returns:
            tuple: 离散化后的状态键
        """
        # 将连续观测值离散化为5个等级
        bins = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # 对整个观测值进行离散化
        discrete_obs = np.digitize(obs, bins) - 1  # 值域变为 [0,1,2,3]
        
        # 为了进一步减少状态空间，可以每隔几个值取样
        # 例如：每3个值取一个样本
        sampled_obs = discrete_obs[::3]
        
        return tuple(sampled_obs)

    def _value_iteration(self):
        """值迭代算法"""
        # 从收集的样本中获取所有唯一状态
        states = set()
        for (state, _) in self.state_transitions.keys():
            states.add(state)
        
        # 初始化值函数
        self.value_function = {state: 0.0 for state in states}
        self.policy = {state: self.action_space[0] for state in states}
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            delta = 0
            for state in states:
                old_value = self.value_function[state]
                
                # 计算每个动作的值
                action_values = {}
                for action in self.action_space:
                    action_key = tuple(np.array(action).flatten())
                    if (state, action_key) in self.state_transitions:
                        # 计算期望值
                        transitions = self.state_transitions[(state, action_key)]
                        expected_value = 0
                        for next_state, reward in transitions:
                            expected_value += (reward + self._discount_factor * self.value_function.get(next_state, 0))
                        expected_value /= len(transitions)
                        action_values[action] = expected_value
                
                # 如果有可用的动作，更新值函数和策略
                if action_values:
                    best_action = max(action_values.keys(), key=lambda a: action_values[a])
                    self.value_function[state] = action_values[best_action]
                    self.policy[state] = best_action
                    
                    delta = max(delta, abs(old_value - self.value_function[state]))
            
            print(f"Iteration {iteration}, Delta: {delta}")
            
            # 检查收敛
            if delta < self._threshold:
                print("Value iteration converged.")
                break
            
            iteration += 1

    def _build_action_space(self) -> List[Tuple[int, float]]:
        """构建离散的动作空间"""
        offloading_targets = range(self._task_offloading_number)
        resource_levels = np.linspace(0, 1, 5)  # 5个离散的资源分配级别
        
        from itertools import product
        action_space = list(product(offloading_targets, resource_levels))
        return action_space






    