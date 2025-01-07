import numpy as np
from typing import List, Dict

class GreedyAgent:
    def __init__(
        self,
        client_vehicle_num: int,
        maximum_task_generation_number: int, 
        maximum_server_vehicle_num: int,
        edge_num: int,
        max_action: int,
        agent_number: int
    ):
        self._client_vehicle_num = client_vehicle_num
        self._maximum_task_generation_number = maximum_task_generation_number
        self._maximum_server_vehicle_num = maximum_server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + maximum_server_vehicle_num + edge_num + 1 # local, vehicles, edges, cloud
        self._max_action = max_action
        self._agent_number = agent_number

    def generate_action(self, obs: List[np.ndarray]) -> np.ndarray:
        """基于贪心策略生成动作
        
        Args:
            obs: 包含每个智能体观测值的列表
            
        Returns:
            actions: 形状为 (agent_num, max_action) 的动作数组
        """
        actions = np.zeros((self._agent_number, self._max_action))
        
        # Task Offloading Agent (agent_index = 0)
        task_offloading_actions = self._generate_task_offloading_actions(obs[0])
        actions[0] = task_offloading_actions
        
        # Resource Allocation Agent (agent_index = 1) 
        resource_allocation_actions = self._generate_resource_allocation_actions(obs[1])
        actions[1] = resource_allocation_actions

        return actions

    def _generate_task_offloading_actions(self, obs: np.ndarray) -> np.ndarray:
        """为任务卸载代理生成贪心动作"""
        actions = np.zeros(self._client_vehicle_num * self._maximum_task_generation_number)
        
        # 解析观测中的队列状态
        queue_states = self._parse_queue_states(obs)
        
        for i in range(self._client_vehicle_num):
            for j in range(self._maximum_task_generation_number):
                idx = i * self._maximum_task_generation_number + j
                
                # 计算每个卸载选项的得分
                scores = self._compute_offloading_scores(queue_states, i)
                
                # 选择得分最高的选项
                best_option = np.argmax(scores)
                
                # 将选择归一化到[0,1]区间
                actions[idx] = best_option / (self._task_offloading_number - 1)
                
        return actions

    def _generate_resource_allocation_actions(self, obs: np.ndarray) -> np.ndarray:
        """为资源分配代理生成贪心动作
        
        Returns:
            actions: 包含所有节点资源分配的数组，按以下顺序排列：
            1. 客户车辆资源分配 (计算资源 + 通信资源)
            2. 服务车辆计算资源分配
            3. 边缘节点计算资源分配
            4. 云服务器计算资源分配
        """
        # 解析观测中的资源需求状态
        resource_demands = self._parse_resource_demands(obs)
        
        # 初始化动作数组
        actions = np.zeros(self._max_action)
        
        # 1. 客户车辆资源分配
        client_vehicle_length = self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2)
        for i in range(self._client_vehicle_num):
            start_idx = i * (self._maximum_task_offloaded_at_client_vehicle_number + 2)
            # 计算资源分配
            compute_demands = resource_demands[start_idx:start_idx + self._maximum_task_offloaded_at_client_vehicle_number]
            if np.sum(compute_demands) > 0:
                actions[start_idx:start_idx + self._maximum_task_offloaded_at_client_vehicle_number] = \
                    compute_demands / np.sum(compute_demands)
            # 通信资源分配 (最后两个元素)
            comm_demands = resource_demands[start_idx + self._maximum_task_offloaded_at_client_vehicle_number:
                                          start_idx + self._maximum_task_offloaded_at_client_vehicle_number + 2]
            if np.sum(comm_demands) > 0:
                actions[start_idx + self._maximum_task_offloaded_at_client_vehicle_number:
                       start_idx + self._maximum_task_offloaded_at_client_vehicle_number + 2] = \
                    comm_demands / np.sum(comm_demands)
        
        # 2. 服务车辆计算资源分配
        server_start = client_vehicle_length
        server_length = self._server_vehicle_num * self._maximum_task_offloaded_at_server_vehicle_number
        for i in range(self._server_vehicle_num):
            start_idx = server_start + i * self._maximum_task_offloaded_at_server_vehicle_number
            end_idx = start_idx + self._maximum_task_offloaded_at_server_vehicle_number
            demands = resource_demands[start_idx:end_idx]
            if np.sum(demands) > 0:
                actions[start_idx:end_idx] = demands / np.sum(demands)
        
        # 3. 边缘节点计算资源分配
        edge_start = server_start + server_length
        edge_length = self._edge_num * self._maximum_task_offloaded_at_edge_node_number
        for i in range(self._edge_num):
            start_idx = edge_start + i * self._maximum_task_offloaded_at_edge_node_number
            end_idx = start_idx + self._maximum_task_offloaded_at_edge_node_number
            demands = resource_demands[start_idx:end_idx]
            if np.sum(demands) > 0:
                actions[start_idx:end_idx] = demands / np.sum(demands)
        
        # 4. 云服务器计算资源分配
        cloud_start = edge_start + edge_length
        cloud_demands = resource_demands[cloud_start:cloud_start + self._maximum_task_offloaded_at_cloud_number]
        if np.sum(cloud_demands) > 0:
            actions[cloud_start:cloud_start + self._maximum_task_offloaded_at_cloud_number] = \
                cloud_demands / np.sum(cloud_demands)
        
        return actions

    def _parse_queue_states(self, obs: np.ndarray) -> Dict:
        """从观测中提取队列状态信息
        
        Args:
            obs: 形状为(self._max_observation,)的观测向量
            
        Returns:
            queue_states: 包含各类队列状态的字典
        """
        # 计算各部分的起始索引
        task_info_size = self._client_vehicle_num * self._maximum_task_generation_number * 2
        lc_queue_start = task_info_size
        v2v_conn_start = lc_queue_start + self._client_vehicle_num
        server_queue_start = v2v_conn_start + self.v2v_pca.n_components_
        v2i_conn_start = server_queue_start + self._server_vehicle_num * 2
        edge_queue_start = v2i_conn_start + self.v2i_pca.n_components_
        cloud_queue_start = edge_queue_start + self._edge_num * 3
        
        queue_states = {
            # 本地计算队列积压
            'local': obs[lc_queue_start:lc_queue_start + self._client_vehicle_num],
            
            # 服务器车辆队列积压 (v2v和vc)
            'v2v': obs[server_queue_start:server_queue_start + self._server_vehicle_num * 2:2],
            'vc': obs[server_queue_start + 1:server_queue_start + self._server_vehicle_num * 2:2],
            
            # 边缘节点队列积压 (v2i, i2i和ec)
            'v2i': obs[edge_queue_start:edge_queue_start + self._edge_num * 3:3],
            'i2i': obs[edge_queue_start + 1:edge_queue_start + self._edge_num * 3:3],
            'ec': obs[edge_queue_start + 2:edge_queue_start + self._edge_num * 3:3],
            
            # 云服务器队列积压 (i2c和cc)
            'i2c': obs[cloud_queue_start],
            'cc': obs[cloud_queue_start + 1],
            
            # 连接状态
            'v2v_conn': obs[v2v_conn_start:v2v_conn_start + self.v2v_pca.n_components_],
            'v2i_conn': obs[v2i_conn_start:v2i_conn_start + self.v2i_pca.n_components_]
        }
    
        return queue_states

    def _parse_resource_demands(self, obs: np.ndarray) -> np.ndarray:
        """从观测中提取资源需求信息
        
        Args:
            obs: 形状为(self._max_observation,)的观测向量
            
        Returns:
            resource_demands: 各节点的资源需求数组，包含：
            1. 客户车辆: 计算资源(tasks) + 通信资源(2)
            2. 服务车辆: 计算资源(tasks)
            3. 边缘节点: 计算资源(tasks)
            4. 云服务器: 计算资源(tasks)
        """
        # 初始化资源需求数组
        resource_demands = np.zeros(self._max_action)
        
        # 获取队列状态
        queue_states = self._parse_queue_states(obs)
        
        # 获取任务信息 (前self._client_vehicle_num * self._maximum_task_generation_number * 2个元素)
        task_info_size = self._client_vehicle_num * self._maximum_task_generation_number * 2
        task_sizes = obs[0:task_info_size:2]  # 数据大小
        task_cycles = obs[1:task_info_size:2]  # CPU需求
        
        current_idx = 0
        
        # 1. 客户车辆资源需求
        for i in range(self._client_vehicle_num):
            # 计算资源需求 - 基于本地队列长度和任务CPU需求
            local_tasks = min(queue_states['local'][i], self._maximum_task_offloaded_at_client_vehicle_number)
            if local_tasks > 0:
                task_demands = task_cycles[i * self._maximum_task_generation_number:
                                        (i + 1) * self._maximum_task_generation_number][:local_tasks]
                resource_demands[current_idx:current_idx + local_tasks] = task_demands
            
            current_idx += self._maximum_task_offloaded_at_client_vehicle_number
            
            # 通信资源需求 - 基于V2V和V2I队列状态
            v2v_demand = np.mean(queue_states['v2v']) if 'v2v' in queue_states else 0
            v2i_demand = np.mean(queue_states['v2i']) if 'v2i' in queue_states else 0
            resource_demands[current_idx] = v2v_demand
            resource_demands[current_idx + 1] = v2i_demand
            current_idx += 2
        
        # 2. 服务车辆资源需求
        for i in range(self._server_vehicle_num):
            # 基于VC队列长度和平均任务CPU需求
            vc_tasks = min(queue_states['vc'][i], self._maximum_task_offloaded_at_server_vehicle_number)
            if vc_tasks > 0:
                avg_task_demand = np.mean(task_cycles)  # 使用平均CPU需求
                resource_demands[current_idx:current_idx + vc_tasks] = avg_task_demand
            current_idx += self._maximum_task_offloaded_at_server_vehicle_number
        
        # 3. 边缘节点资源需求
        for i in range(self._edge_num):
            # 基于EC队列长度和平均任务CPU需求
            ec_tasks = min(queue_states['ec'][i], self._maximum_task_offloaded_at_edge_node_number)
            if ec_tasks > 0:
                avg_task_demand = np.mean(task_cycles)
                resource_demands[current_idx:current_idx + ec_tasks] = avg_task_demand
            current_idx += self._maximum_task_offloaded_at_edge_node_number
        
        # 4. 云服务器资源需求
        # 基于CC队列长度和平均任务CPU需求
        cc_tasks = min(queue_states['cc'], self._maximum_task_offloaded_at_cloud_number)
        if cc_tasks > 0:
            avg_task_demand = np.mean(task_cycles)
            resource_demands[current_idx:current_idx + cc_tasks] = avg_task_demand
        
        return np.maximum(resource_demands, 0)  # 确保需求非负

    def _compute_offloading_scores(self, queue_states: Dict, client_idx: int) -> np.ndarray:
        """计算每个卸载选项的得分"""
        scores = np.zeros(self._task_offloading_number)
        
        # 本地计算得分
        scores[0] = 1 / (1 + queue_states['local'][client_idx])
        
        # 车辆计算得分
        for i in range(self._maximum_server_vehicle_num):
            scores[1 + i] = 1 / (1 + queue_states['vehicle'][i])
            
        # 边缘节点得分
        for i in range(self._edge_num):
            scores[1 + self._maximum_server_vehicle_num + i] = 1 / (1 + queue_states['edge'][i])
            
        # 云计算得分
        scores[-1] = 1 / (1 + queue_states['cloud'])
        
        return scores