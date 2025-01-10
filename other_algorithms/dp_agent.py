import numpy as np
from typing import List, Dict, Tuple, Union
from itertools import product

class DPAgent:
    def __init__(
        self,
        client_vehicle_num: int,
        maximum_task_generation_number_of_vehicles: int,
        maximum_server_vehicle_num: int,
        server_vehicle_num: int,
        edge_num: int,
        max_action: int,
        maximum_task_offloaded_at_client_vehicle_number: int,
        maximum_task_offloaded_at_server_vehicle_number: int,
        maximum_task_offloaded_at_edge_node_number: int,
        maximum_task_offloaded_at_cloud_number: int,
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
        self._server_vehicle_num = server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + maximum_server_vehicle_num + edge_num + 1
        self._max_action = max_action
        self._maximum_task_offloaded_at_client_vehicle_number = maximum_task_offloaded_at_client_vehicle_number
        self._maximum_task_offloaded_at_server_vehicle_number = maximum_task_offloaded_at_server_vehicle_number
        self._maximum_task_offloaded_at_edge_node_number = maximum_task_offloaded_at_edge_node_number
        self._maximum_task_offloaded_at_cloud_number = maximum_task_offloaded_at_cloud_number
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

    def generate_action(self, obs: List[np.ndarray]) -> np.ndarray:
        """生成动作，并在预热后进行值迭代"""
        self._step_count += 1
        
        # 生成动作
        if not self._use_dp or self._step_count < self._warmup_steps:
            # 使用启发式策略
            actions = self._generate_random_actions(obs)
        else:
            if self._step_count >= self._warmup_steps:
                print("Starting value iteration...")
                self._value_iteration()
            # 使用优化后的策略
            actions = self._generate_optimal_actions(obs)
        
        return actions
    
    def store_transition(self, obs: List[np.ndarray], current_state: tuple, actions: np.ndarray, env_info: dict = None):
        """存储状态转移样本"""
        # 如果有环境信息，存储转移样本
        if env_info is not None and self._step_count < self._warmup_steps:
            reward = env_info.get('reward', 0)
            next_obs = env_info.get('next_obs', obs)
            
            # 确保 current_state 是元组类型
            if isinstance(current_state, np.ndarray):
                current_state = tuple(current_state)
            
            # 获取下一个状态的键
            next_state = self._get_state_key(next_obs[0])
            # 获取动作的键
            action_key = self._get_action_key(actions)
            
            # 存储转移样本
            state_action_key = (current_state, action_key)
            if state_action_key not in self.state_transitions:
                self.state_transitions[state_action_key] = []
            self.state_transitions[state_action_key].append((next_state, reward))

    def _generate_random_actions(self, obs: List[np.ndarray]) -> np.ndarray:
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
            optimal_action_index = self.policy[state_key]  # 这是一个整数索引
            optimal_action = self.action_space[optimal_action_index]  # 获取实际的动作元组
            
            # Task offloading agent
            task_offload_actions = []
            task_offloading_number = 1 + self._maximum_server_vehicle_num + self._edge_num + 1
            for client_vehicle_index in range(self._client_vehicle_num):
                for task_index in range(self._maximum_task_generation_number_of_vehicles):
                    action_index = client_vehicle_index * self._maximum_task_generation_number_of_vehicles + task_index
                    if action_index < len(self.action_space):
                        # 生成[0,1]范围内的连续值
                        base_value = self.action_space[action_index][1]  # 使用动作空间中的值
                        continuous_value = np.random.uniform(
                            base_value,
                            min(1.0, base_value + 1.0/(task_offloading_number-1))
                        )
                        task_offload_actions.append(continuous_value)
            
            # Resource allocation agent
            resource_actions = []
            # 客户车辆资源
            start_index = len(task_offload_actions)
            for client_vehicle_index in range(self._client_vehicle_num):
                # 计算资源分配
                for task_index in range(self._maximum_task_offloaded_at_client_vehicle_number):
                    action_index = start_index + len(resource_actions)
                    if action_index < len(self.action_space):
                        base_value = self.action_space[action_index][1]
                        continuous_value = np.random.uniform(
                            base_value,
                            min(1.0, base_value + 0.25)
                        )
                        resource_actions.append(continuous_value)
                # V2V和V2I通信资源
                for _ in range(2):
                    action_index = start_index + len(resource_actions)
                    if action_index < len(self.action_space):
                        base_value = self.action_space[action_index][1]
                        continuous_value = np.random.uniform(
                            base_value,
                            min(1.0, base_value + 0.25)
                        )
                        resource_actions.append(continuous_value)
            
            # 服务车辆资源
            for server_vehicle_index in range(self._server_vehicle_num):
                for task_index in range(self._maximum_task_offloaded_at_server_vehicle_number):
                    action_index = start_index + len(resource_actions)
                    if action_index < len(self.action_space):
                        base_value = self.action_space[action_index][1]
                        continuous_value = np.random.uniform(
                            base_value,
                            min(1.0, base_value + 0.25)
                        )
                        resource_actions.append(continuous_value)
            
            # 边缘节点资源
            for edge_node_index in range(self._edge_num):
                for task_index in range(self._maximum_task_offloaded_at_edge_node_number):
                    action_index = start_index + len(resource_actions)
                    if action_index < len(self.action_space):
                        base_value = self.action_space[action_index][1]
                        continuous_value = np.random.uniform(
                            base_value,
                            min(1.0, base_value + 0.25)
                        )
                        resource_actions.append(continuous_value)
            
            # 云节点资源
            for task_index in range(self._maximum_task_offloaded_at_cloud_number):
                action_index = start_index + len(resource_actions)
                if action_index < len(self.action_space):
                    base_value = self.action_space[action_index][1]
                    continuous_value = np.random.uniform(
                        base_value,
                        min(1.0, base_value + 0.25)
                    )
                    resource_actions.append(continuous_value)
            
            # 填充到actions数组
            actions[0][:len(task_offload_actions)] = task_offload_actions
            actions[1][:len(resource_actions)] = resource_actions
            
        else:
            # 对于未见过的状态使用启发式策略
            return self._generate_random_actions(obs)
        
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
        
    def _get_action_key(self, actions: Union[np.ndarray, Tuple[int, float]]) -> tuple:
        """将连续动作值离散化为动作键
        
        Args:
            actions: 动作值数组或(索引,值)元组
            
        Returns:
            tuple: 离散化后的动作键
        """
        # 如果输入是(索引,值)元组，直接返回索引作为键
        if isinstance(actions, tuple):
            return (actions[0],)
        
        # 否则处理动作数组
        # 将连续动作值离散化为5个等级
        bins = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # 对整个动作值进行离散化
        discrete_actions = []
        for agent_actions in actions:
            # 对每个智能体的动作进行离散化
            agent_discrete = np.digitize(agent_actions, bins) - 1  # 值域变为 [0,1,2,3]
            # 为了进一步减少动作空间，可以每隔几个值取样
            sampled_actions = agent_discrete[::3]
            discrete_actions.extend(sampled_actions)
        
        # 转换为元组使其可哈希
        return tuple(discrete_actions)

    def _value_iteration(self):
        """值迭代算法，支持间隔性迭代"""
        # 从收集的样本中获取所有唯一状态
        states = set()
        for (state, _) in self.state_transitions.keys():
            states.add(state)
        
        # 只在第一次运行时初始化值函数和策略
        if not hasattr(self, '_value_iteration_initialized'):
            self.value_function = {state: 0.0 for state in states}
            self.policy = {state: 0 for state in states}  # 存储索引而不是动作元组
            self._value_iteration_initialized = True
            self._last_iteration_step = self._step_count
            self._total_iterations = 0
        
        # 检查是否需要进行值迭代
        iteration_interval = max(self._warmup_steps // 10, 100)
        if self._step_count - self._last_iteration_step < iteration_interval:
            return
        
        print(f"Continuing value iteration at step {self._step_count}")
        print(f"Total iterations so far: {self._total_iterations}")
        
        # 执行有限次数的值迭代
        max_iterations_per_update = 10
        iteration = 0
        
        while iteration < max_iterations_per_update:
            delta = 0
            for state in states:
                old_value = self.value_function[state]
                
                # 计算每个动作的值
                action_values = {}
                for action_idx, action in enumerate(self.action_space):  # 使用enumerate获取索引
                    action_key = self._get_action_key(action)
                    if (state, action_key) in self.state_transitions:
                        transitions = self.state_transitions[(state, action_key)]
                        expected_value = 0
                        for next_state, reward in transitions:
                            expected_value += (reward + self._discount_factor * self.value_function.get(next_state, 0))
                        expected_value /= len(transitions)
                        action_values[action_idx] = expected_value  # 存储索引而不是动作本身
                
                # 如果有可用的动作，更新值函数和策略
                if action_values:
                    best_action_idx = max(action_values.keys(), key=lambda idx: action_values[idx])
                    self.value_function[state] = action_values[best_action_idx]
                    self.policy[state] = best_action_idx  # 存储最佳动作的索引
                    
                    delta = max(delta, abs(old_value - self.value_function[state]))
            
            print(f"Iteration {self._total_iterations + iteration + 1}, Delta: {delta}")
            
            # 检查收敛
            if delta < self._threshold:
                print("Value iteration converged.")
                break
            
            iteration += 1
        
        # 更新迭代计数和最后迭代步数
        self._total_iterations += iteration
        self._last_iteration_step = self._step_count

    def _build_action_space(self) -> List[Tuple[int, float]]:
        """构建离散的动作空间
        Returns:
            List[Tuple[int, float]]: 动作空间列表，每个元素为(动作索引, 动作值)的元组
        """
        actions = []
        
        # Task offloading agent actions
        task_offloading_number = 1 + self._maximum_server_vehicle_num + self._edge_num + 1
        for client_vehicle_index in range(self._client_vehicle_num):
            for task_index in range(self._maximum_task_generation_number_of_vehicles):
                # 对每个任务的卸载决策离散化为task_offloading_number个级别
                for offload_level in range(task_offloading_number):
                    actions.append((len(actions), offload_level / (task_offloading_number - 1)))
        
        # Resource allocation agent actions
        # 客户车辆资源分配
        for client_vehicle_index in range(self._client_vehicle_num):
            # 计算资源分配
            for task_index in range(self._maximum_task_offloaded_at_client_vehicle_number):
                for resource_level in range(5):  # 5个离散的资源分配级别
                    actions.append((len(actions), resource_level / 4))
            # V2V和V2I通信资源分配
            for _ in range(2):
                for resource_level in range(5):
                    actions.append((len(actions), resource_level / 4))
        
        # 服务车辆计算资源分配
        for server_vehicle_index in range(self._server_vehicle_num):
            for task_index in range(self._maximum_task_offloaded_at_server_vehicle_number):
                for resource_level in range(5):
                    actions.append((len(actions), resource_level / 4))
        
        # 边缘节点计算资源分配
        for edge_node_index in range(self._edge_num):
            for task_index in range(self._maximum_task_offloaded_at_edge_node_number):
                for resource_level in range(5):
                    actions.append((len(actions), resource_level / 4))
        
        # 云节点计算资源分配
        for task_index in range(self._maximum_task_offloaded_at_cloud_number):
            for resource_level in range(5):
                actions.append((len(actions), resource_level / 4))
            
        return actions






    