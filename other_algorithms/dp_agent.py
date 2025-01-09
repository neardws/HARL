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
            if self._step_count == self._warmup_steps:
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
        """使用启发式策略生成动作"""
        actions = np.zeros((self._agent_number, self._max_action))
        actions[0] = self._get_offloading_actions(obs[0])
        actions[1] = self._get_resource_allocation_actions(obs[1])
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
        """将观测转换为状态键"""
        # 只使用任务相关的信息作为状态
        task_info_size = self._client_vehicle_num * self._maximum_task_generation_number_of_vehicles * 2
        task_info = obs[:task_info_size]
        return tuple(task_info)

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

    def _get_offloading_actions(self, obs: np.ndarray) -> np.ndarray:
        """生成任务卸载动作
        
        Args:
            obs: 观测值，其中任务信息包含 [task_size, required_cpu_cycles]
        
        Returns:
            action: 任务卸载动作，shape为(self._max_action,)
                   包含self._client_vehicle_num * self._maximum_task_generation_number_of_vehicles个有效元素
        """
        action = np.zeros(self._max_action)
        
        # 获取任务信息
        task_info_size = self._client_vehicle_num * self._maximum_task_generation_number_of_vehicles * 2
        task_info = obs[:task_info_size].reshape(
            self._client_vehicle_num, 
            self._maximum_task_generation_number_of_vehicles, 
            2  # [task_size, required_cpu_cycles]
        )
        
        # 为每个客户车辆的每个任务生成卸载决策
        action_idx = 0
        for vehicle_idx in range(self._client_vehicle_num):
            for task_idx in range(self._maximum_task_generation_number_of_vehicles):
                task_size = task_info[vehicle_idx, task_idx, 0]
                cpu_cycles = task_info[vehicle_idx, task_idx, 1]
                
                # 只为有效任务生成卸载决策
                if task_size > 0:
                    # 基于任务特征选择卸载目标
                    if task_size < 0.3 and cpu_cycles < 0.3:  # 小任务且计算需求低
                        offload_target = 0  # 本地处理
                    elif task_size < 0.7:  # 中等大小任务
                        if cpu_cycles < 0.5:  # 计算需求适中
                            offload_target = np.random.randint(1, self._maximum_server_vehicle_num + 1)  # 选择服务器车辆
                        else:  # 计算需求较高
                            offload_target = self._maximum_server_vehicle_num + np.random.randint(self._edge_num)  # 选择边缘节点
                    else:  # 大任务或计算需求高
                        if cpu_cycles > 0.7:  # 计算需求很高
                            offload_target = self._task_offloading_number - 1  # 选择云服务器
                        else:  # 计算需求适中
                            offload_target = self._maximum_server_vehicle_num + np.random.randint(self._edge_num)  # 选择边缘节点
                    
                    # 将离散动作转换为连续动作
                    action[action_idx] = offload_target / (self._task_offloading_number - 1)
                
                action_idx += 1
        
        return action

    def _get_resource_allocation_actions(self, obs: np.ndarray) -> np.ndarray:
        """生成资源分配动作"""
        action = np.zeros(self._max_action)
        
        # 简单的负载均衡策略
        action[0] = 0.5  # 平均分配资源
        
        return action

    def _build_action_space(self) -> List[Tuple[int, float]]:
        """构建离散的动作空间"""
        offloading_targets = range(self._task_offloading_number)
        resource_levels = np.linspace(0, 1, 5)  # 5个离散的资源分配级别
        
        from itertools import product
        action_space = list(product(offloading_targets, resource_levels))
        return action_space

    def _build_state_space(self) -> List[Tuple]:
        """构建状态空间
        
        Returns:
            states: 所有可能状态的列表
        """
        # 定义每个队列可能的离散值数量
        num_discrete_levels = 10  # 将[0,1]区间划分为5个离散等级
        discrete_values = np.linspace(0, 1, num_discrete_levels)
        
        # 计算各类队列的数量
        num_local_queues = 1  # 本地队列
        num_v2v_queues = self._maximum_server_vehicle_num  # v2v队列
        num_vc_queues = self._maximum_server_vehicle_num   # vc队列
        num_v2i_queues = self._edge_num  # v2i队列
        num_i2i_queues = self._edge_num  # i2i队列
        num_ec_queues = self._edge_num   # ec队列
        num_cloud_queues = 2  # i2c和cc队列
        
        # 计算总的队列数量
        total_queues = (num_local_queues + 
                       num_v2v_queues + 
                       num_vc_queues + 
                       num_v2i_queues + 
                       num_i2i_queues + 
                       num_ec_queues + 
                       num_cloud_queues)
        
        # 为了避免状态空间过大，使用代表性状态组合
        max_states = 10000  # 限制状态空间大小
        
        states = []
        
        # 生成代表性状态组合
        num_samples = min(max_states, num_discrete_levels ** total_queues)
        
        for _ in range(num_samples):
            # 随机生成一个状态
            state = []
            
            # 本地队列
            state.append(np.random.choice(discrete_values))
            
            # 服务器车辆队列 (v2v和vc)
            for _ in range(num_v2v_queues):
                state.append(np.random.choice(discrete_values))
            for _ in range(num_vc_queues):
                state.append(np.random.choice(discrete_values))
            
            # 边缘节点队列 (v2i, i2i和ec)
            for _ in range(num_v2i_queues):
                state.append(np.random.choice(discrete_values))
            for _ in range(num_i2i_queues):
                state.append(np.random.choice(discrete_values))
            for _ in range(num_ec_queues):
                state.append(np.random.choice(discrete_values))
            
            # 云服务器队列 (i2c和cc)
            for _ in range(num_cloud_queues):
                state.append(np.random.choice(discrete_values))
            
            states.append(tuple(state))
        
        # 添加一些特殊状态
        special_states = [
            # 全空状态
            tuple([0.0] * total_queues),
            # 全满状态
            tuple([1.0] * total_queues),
            # 本地队列满其他为空
            tuple([1.0] + [0.0] * (total_queues - 1)),
            # 边缘节点队列满其他为空
            tuple([0.0] * (1 + 2 * num_v2v_queues) + 
                 [1.0] * (3 * num_v2i_queues) + 
                 [0.0] * num_cloud_queues),
            # 云服务器队列满其他为空
            tuple([0.0] * (total_queues - 2) + [1.0, 1.0])
        ]
        
        # 将特殊状态添加到状态空间
        states.extend(special_states)
        
        # 去除重复状态
        states = list(set(states))
        
        return states

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
        normalized_action = resource_allocation_action / (len(self.action_space) - 1)
        return normalized_action

    def _map_obs_to_state(self, obs: np.ndarray) -> Tuple:
        """将观测值映射到状态
        
        Args:
            obs: 形状为(self._max_observation,)的观测向量
            
        Returns:
            state: 对应的状态元组，包含原始的队列状态值[0,1]
        """
        # 计算各部分的起始索引
        task_info_size = self._client_vehicle_num * self._maximum_task_generation_number_of_vehicles * 2
        lc_queue_start = task_info_size
        v2v_conn_start = lc_queue_start + self._client_vehicle_num
        server_queue_start = v2v_conn_start + self._v2v_n_components
        v2i_conn_start = server_queue_start + self._server_vehicle_num * 2
        edge_queue_start = v2i_conn_start + self._v2i_n_components
        cloud_queue_start = edge_queue_start + self._edge_num * 3
        
        # 直接使用原始队列状态值
        lc_queue = obs[lc_queue_start]
        
        # 服务器车辆队列 (v2v和vc)
        v2v_queues = tuple(obs[server_queue_start:server_queue_start + self._server_vehicle_num * 2:2])
        vc_queues = tuple(obs[server_queue_start + 1:server_queue_start + self._server_vehicle_num * 2:2])
        
        # 边缘节点队列 (v2i, i2i和ec)
        v2i_queues = tuple(obs[edge_queue_start:edge_queue_start + self._edge_num * 3:3])
        i2i_queues = tuple(obs[edge_queue_start + 1:edge_queue_start + self._edge_num * 3:3])
        ec_queues = tuple(obs[edge_queue_start + 2:edge_queue_start + self._edge_num * 3:3])
        
        # 云服务器队列 (i2c和cc)
        i2c_queue = obs[cloud_queue_start]
        cc_queue = obs[cloud_queue_start + 1]
        
        # 组合所有状态为一个元组
        state = (lc_queue,) + v2v_queues + vc_queues + v2i_queues + i2i_queues + ec_queues + (i2c_queue, cc_queue)
        return state

    def _get_all_possible_actions(self) -> List[Tuple]:
        """
        获取所有可能的动作组合。

        Returns:
            actions: 动作的列表。
        """
        return self.action_space



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

    def _get_transition_prob(self, state: Tuple, action: int, next_state: Tuple) -> float:
        """计算状态转移概率
        
        Args:
            state: 当前状态元组，包含[0,1]范围内的队列状态值
            action: 选择的动作
            next_state: 下一个状态元组
            
        Returns:
            transition_prob: 状态转移概率
        """
        # 使用高斯分布来模拟连续状态转移
        transition_prob = 1.0
        
        # 分解状态元组中的各个队列状态
        current_queues = np.array(state)
        next_queues = np.array(next_state)
        
        # 根据动作类型计算期望的队列变化
        expected_changes = self._compute_expected_changes(current_queues, action)
        expected_next_queues = np.clip(current_queues + expected_changes, 0, 1)
        
        # 计算转移概率（使用高斯分布）
        sigma = 0.1  # 标准差可以根据实际情况调整
        transition_prob = np.prod(np.exp(-((next_queues - expected_next_queues) ** 2) / (2 * sigma ** 2)))
        
        return transition_prob

    def _compute_expected_changes(self, current_queues: np.ndarray, action: int) -> np.ndarray:
        """计算期望的队列状态变化
        
        Args:
            current_queues: 当前队列状态数组
            action: 选择的动作
            
        Returns:
            expected_changes: 期望的队列状态变化
        """
        num_queues = len(current_queues)
        expected_changes = np.zeros(num_queues)
        
        # 根据动作类型计算期望变化
        if action == 0:  # 本地处理
            expected_changes[0] = -0.2  # 本地队列减少
        elif action <= self._maximum_server_vehicle_num:  # 卸载到服务器车辆
            server_idx = action - 1
            v2v_idx = 1 + server_idx  # v2v队列索引
            vc_idx = 1 + self._maximum_server_vehicle_num + server_idx  # vc队列索引
            expected_changes[0] = -0.2  # 本地队列减少
            expected_changes[v2v_idx] = 0.15  # v2v队列增加
            expected_changes[vc_idx] = 0.15  # vc队列增加
        elif action <= self._maximum_server_vehicle_num + self._edge_num:  # 卸载到边缘节点
            edge_idx = action - 1 - self._maximum_server_vehicle_num
            v2i_start_idx = 1 + 2 * self._maximum_server_vehicle_num
            expected_changes[0] = -0.2  # 本地队列减少
            expected_changes[v2i_start_idx + edge_idx] = 0.15  # v2i队列增加
            expected_changes[v2i_start_idx + self._edge_num + edge_idx] = 0.1  # ec队列增加
        else:  # 卸载到云服务器
            cloud_start_idx = -2
            expected_changes[0] = -0.2  # 本地队列减少
            expected_changes[cloud_start_idx] = 0.15  # i2c队列增加
            expected_changes[cloud_start_idx + 1] = 0.1  # cc队列增加
        
        return expected_changes

    def _get_reward(self, state: Tuple, action: int, next_state: Tuple) -> float:
        """计算奖励值
        
        Args:
            state: 当前状态元组
            action: 选择的动作
            next_state: 下一个状态元组
            
        Returns:
            reward: 奖励值
        """
        current_queues = np.array(state)
        next_queues = np.array(next_state)
        
        # 计算队列负载变化
        queue_changes = next_queues - current_queues
        
        # 根据不同类型的队列赋予不同权重
        weights = self._get_queue_weights()
        weighted_changes = queue_changes * weights
        
        # 负载减少得到正奖励，增加得到负奖励
        reward = -np.sum(weighted_changes)
        
        # 添加队列平衡项
        balance_penalty = self._calculate_balance_penalty(next_queues)
        reward -= balance_penalty
        
        return reward

    def _get_queue_weights(self) -> np.ndarray:
        """获取不同队列的权重"""
        num_queues = (1 +  # 本地队列
                     2 * self._maximum_server_vehicle_num +  # v2v和vc队列
                     3 * self._edge_num +  # v2i, i2i和ec队列
                     2)  # i2c和cc队列
        
        weights = np.ones(num_queues)
        
        # 可以根据实际情况调整不同队列的权重
        weights[0] = 1.2  # 本地队列权重
        
        # v2v和vc队列权重
        v2v_start = 1
        weights[v2v_start:v2v_start + self._maximum_server_vehicle_num] = 1.0
        weights[v2v_start + self._maximum_server_vehicle_num:v2v_start + 2 * self._maximum_server_vehicle_num] = 1.1
        
        # 边缘节点相关队列权重
        edge_start = v2v_start + 2 * self._maximum_server_vehicle_num
        weights[edge_start:edge_start + self._edge_num] = 0.9  # v2i
        weights[edge_start + self._edge_num:edge_start + 2 * self._edge_num] = 1.0  # i2i
        weights[edge_start + 2 * self._edge_num:edge_start + 3 * self._edge_num] = 1.1  # ec
        
        # 云服务器相关队列权重
        weights[-2:] = [0.8, 1.0]  # i2c和cc
        
        return weights

    def _calculate_balance_penalty(self, queues: np.ndarray) -> float:
        """计算负载均衡惩罚项"""
        # 计算各类队列的方差
        server_queues = queues[1:1 + 2 * self._maximum_server_vehicle_num]
        edge_queues = queues[1 + 2 * self._maximum_server_vehicle_num:-2]
        cloud_queues = queues[-2:]
        
        # 使用方差来衡量负载不均衡程度
        server_variance = np.var(server_queues) if len(server_queues) > 0 else 0
        edge_variance = np.var(edge_queues) if len(edge_queues) > 0 else 0
        cloud_variance = np.var(cloud_queues) if len(cloud_queues) > 0 else 0
        
        # 总的不平衡惩罚
        balance_penalty = 0.1 * (server_variance + edge_variance + cloud_variance)
        
        return balance_penalty 