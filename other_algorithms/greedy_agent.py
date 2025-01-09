import numpy as np
from typing import List, Dict

class GreedyAgent:
    def __init__(
        self,
        client_vehicle_num: int,
        maximum_task_generation_number: int, 
        maximum_server_vehicle_num: int,
        server_vehicle_num: int,
        edge_num: int,
        max_action: int,
        agent_number: int,
        maximum_task_offloaded_at_client_vehicle_number: int,
        maximum_task_offloaded_at_server_vehicle_number: int,
        maximum_task_offloaded_at_edge_node_number: int,
        maximum_task_offloaded_at_cloud_number: int,
        v2v_n_components: int,
        v2i_n_components: int
    ):
        self._client_vehicle_num = client_vehicle_num
        self._maximum_task_generation_number = maximum_task_generation_number
        self._maximum_server_vehicle_num = maximum_server_vehicle_num
        self._server_vehicle_num = server_vehicle_num
        self._edge_num = edge_num
        self._task_offloading_number = 1 + maximum_server_vehicle_num + edge_num + 1 # local, vehicles, edges, cloud
        self._max_action = max_action
        self._agent_number = agent_number
        self._maximum_task_offloaded_at_client_vehicle_number = maximum_task_offloaded_at_client_vehicle_number
        self._maximum_task_offloaded_at_server_vehicle_number = maximum_task_offloaded_at_server_vehicle_number
        self._maximum_task_offloaded_at_edge_node_number = maximum_task_offloaded_at_edge_node_number
        self._maximum_task_offloaded_at_cloud_number = maximum_task_offloaded_at_cloud_number
        self._v2v_n_components = v2v_n_components
        self._v2i_n_components = v2i_n_components

    def generate_action(
        self, 
        obs: List[np.ndarray],
        v2v_connections: np.ndarray,
        v2i_connections: np.ndarray
    ) -> np.ndarray:
        """基于贪心策略生成动作
        
        Args:
            obs: 包含每个智能体观测值的列表
            
        Returns:
            actions: 形状为 (agent_num, max_action) 的动作数组
        """
        actions = np.zeros((self._agent_number, self._max_action))
        
        # Task Offloading Agent (agent_index = 0)
        task_offloading_actions, task_offloading_actions_dict = self._generate_task_offloading_actions(obs[0], v2v_connections, v2i_connections)

        actions[0] = task_offloading_actions
        
        # Resource Allocation Agent (agent_index = 1) 
        resource_allocation_actions = self._generate_resource_allocation_actions(obs[1], v2v_connections, v2i_connections, task_offloading_actions_dict)
        actions[1] = resource_allocation_actions

        return actions

    def _generate_task_offloading_actions(
        self, 
        obs: np.ndarray, 
        v2v_connections: np.ndarray, 
        v2i_connections: np.ndarray
    ) -> np.ndarray:
        """为任务卸载代理生成贪心动作"""
        actions = np.zeros(self._client_vehicle_num * self._maximum_task_generation_number)
        
        # 解析观测中的队列状态
        queue_states = self._parse_queue_states(obs, v2v_connections, v2i_connections)

        task_offloading_actions_dict = {}

        for i in range(self._client_vehicle_num):
            for j in range(self._maximum_task_generation_number):
                idx = i * self._maximum_task_generation_number + j
                
                # 计算每个卸载选项的得分
                scores = self._compute_offloading_scores(queue_states, i)

                # print("scores: ", scores)
                
                # 选择得分最高的选项
                best_option = np.argmax(scores)

                if best_option == 0:
                    task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Local"
                elif best_option >= 1 and best_option <= self._maximum_server_vehicle_num:
                    tag = 0
                    flag = False
                    for server_vehicle_index in range(self._server_vehicle_num):
                        if v2v_connections[i][server_vehicle_index] == 1:
                            tag += 1
                            if best_option == tag:
                                flag = True
                                task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Server Vehicle " + str(server_vehicle_index)
                                break
                    if not flag:
                        task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Local"
                    task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Server Vehicle " + str(best_option - 1)
                elif best_option >= 1 + self._maximum_server_vehicle_num and best_option < 1 + self._maximum_server_vehicle_num + self._edge_num:
                    task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Edge Node " + str(best_option - 1 - self._maximum_server_vehicle_num)
                else:
                    task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)] = "Cloud"

                # 将选择归一化到[0,1]区间
                actions[idx] = best_option / (self._task_offloading_number - 1)

        # 填充action 到self._max_action
        actions = np.pad(actions, (0, self._max_action - len(actions)), mode='constant', constant_values=0)
        # print("actions: ", actions)
        return actions, task_offloading_actions_dict

    def _generate_resource_allocation_actions(
        self, 
        obs: np.ndarray, 
        v2v_connections: np.ndarray, 
        v2i_connections: np.ndarray,
        task_offloading_actions_dict: Dict
    ) -> np.ndarray:
        """为资源分配代理生成贪心动作
        
        Returns:
            actions: 包含所有节点资源分配的数组，按以下顺序排列：
            1. 客户车辆资源分配 (计算资源 + 通信资源)
            2. 服务车辆计算资源分配
            3. 边缘节点计算资源分配
            4. 云服务器计算资源分配
        """
        # 解析观测中的资源需求状态
        resource_demands = self._parse_resource_demands(obs, v2v_connections, v2i_connections, task_offloading_actions_dict)

        print("resource_demands: ", resource_demands)
        
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

    def _parse_queue_states(
        self, 
        obs: np.ndarray, 
        v2v_connections: np.ndarray, 
        v2i_connections: np.ndarray
    ) -> Dict:
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
        server_queue_start = v2v_conn_start + self._v2v_n_components
        v2i_conn_start = server_queue_start + self._server_vehicle_num * 2
        edge_queue_start = v2i_conn_start + self._v2i_n_components
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
            'v2v_conn': v2v_connections,
            'v2i_conn': v2i_connections
        }
    
        return queue_states

    def _parse_resource_demands(
        self, 
        obs: np.ndarray, 
        v2v_connections: np.ndarray, 
        v2i_connections: np.ndarray,
        task_offloading_actions_dict: Dict
    ) -> np.ndarray:
        """从观测中提取资源需求信息"""
        # 初始化资源需求数组

        resource_demands = np.zeros(self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2) + \
                                    self._server_vehicle_num * self._maximum_task_offloaded_at_server_vehicle_number + \
                                    self._edge_num * self._maximum_task_offloaded_at_edge_node_number + \
                                    self._maximum_task_offloaded_at_cloud_number)
        
        # 获取任务信息
        task_info_size = int(self._client_vehicle_num * self._maximum_task_generation_number * 2)  # 确保是整数
        task_sizes = obs[0:task_info_size:2]  # 数据大小
        task_cycles = obs[1:task_info_size:2]  # CPU需求

        # reshape task_sizes and task_cycles
        task_sizes = np.array(task_sizes)
        task_cycles = np.array(task_cycles)
        task_sizes = task_sizes.reshape(self._client_vehicle_num, self._maximum_task_generation_number)
        task_cycles = task_cycles.reshape(self._client_vehicle_num, self._maximum_task_generation_number)
        
        # 根据任务卸载决策，计算资源需求 TODO: 还是有问题
        for i in range(self._client_vehicle_num):
            for j in range(self._maximum_task_generation_number):
                task_offloading_action = task_offloading_actions_dict["client_vehicle_" + str(i) + "_task_" + str(j)]
                if task_offloading_action == "Local":
                    resource_demands[i * (self._maximum_task_offloaded_at_client_vehicle_number + 2) + j] += task_cycles[i][j] * task_sizes[i][j]
                elif task_offloading_action.startswith("Server Vehicle"):
                    server_vehicle_index = int(task_offloading_action.split(" ")[2])
                    server_start_idx = self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2)
                    resource_demands[server_start_idx + server_vehicle_index * self._maximum_task_offloaded_at_server_vehicle_number + j] += task_cycles[i][j] * task_sizes[i][j]        
                elif task_offloading_action.startswith("Edge Node"):
                    edge_node_index = int(task_offloading_action.split(" ")[2])
                    edge_start_idx = self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2) + \
                                   self._server_vehicle_num * self._maximum_task_offloaded_at_server_vehicle_number
                    resource_demands[edge_start_idx + edge_node_index * self._maximum_task_offloaded_at_edge_node_number + j] += task_cycles[i][j] * task_sizes[i][j]
                elif task_offloading_action.startswith("Cloud"):
                    resource_demands[-1] += task_cycles[i][j] * task_sizes[i][j]

        # 根据V2V和V2I队列状态，计算通信资源需求
        v2v_demand = np.mean(v2v_connections)
        v2i_demand = np.mean(v2i_connections)
        # 为每个客户车辆分配通信资源需求
        for i in range(self._client_vehicle_num):
            # 计算每个车辆的通信资源起始索引位置
            start_idx = i * (self._maximum_task_offloaded_at_client_vehicle_number + 2) + self._maximum_task_offloaded_at_client_vehicle_number
            # 为每个车辆分配V2V和V2I通信资源
            resource_demands[start_idx] = v2v_demand
            resource_demands[start_idx + 1] = v2i_demand

        return np.maximum(resource_demands, 0)  # 确保需求非负


    def _compute_offloading_scores(self, queue_states: Dict, client_idx: int) -> np.ndarray:
        """计算每个卸载选项的得分
        
        考虑因素:
        1. 本地处理: 只需考虑本地处理队列
        2. 服务车辆: 需考虑V2V通信队列和服务车辆处理队列
        3. 边缘服务器: 需考虑V2I通信队列、I2I传输队列和边缘处理队列
        4. 云服务器: 需考虑V2I通信队列、I2I传输队列、I2C通信队列和云处理队列
        """
        scores = np.zeros(self._task_offloading_number)
        
        # 1. 本地计算得分
        scores[0] = 1 / (1 + queue_states['local'][client_idx])
        
        # 2. 服务车辆计算得分
        v2v_connections = queue_states['v2v_conn']
        index = 0
        for i in range(self._server_vehicle_num):
            if i < len(queue_states['v2v']):  # 确保索引有效
                # 考虑V2V通信队列和服务车辆处理队列
                v2v_score = 1 / (1 + queue_states['v2v'][i])
                vc_score = 1 / (1 + queue_states['vc'][i])
                # 考虑连接状态
                conn_score = v2v_connections[client_idx][i]
                if conn_score > 0:
                    scores[1 + index] = (v2v_score + vc_score) * conn_score
                    index += 1
                    if index >= self._maximum_server_vehicle_num:
                        break
            
        # 3. 边缘服务器计算得分
        v2i_connections = queue_states['v2i_conn']
        for i in range(self._edge_num):
            if i < len(queue_states['v2i']):  # 确保索引有效
                # 考虑V2I通信队列、I2I传输队列和边缘处理队列
                v2i_score = 1 / (1 + queue_states['v2i'][i])
                i2i_score = 1 / (1 + queue_states['i2i'][i])
                ec_score = 1 / (1 + queue_states['ec'][i])
                # 考虑连接状态
                conn_score = v2i_connections[client_idx][i]
                scores[1 + self._maximum_server_vehicle_num + i] = (v2i_score + i2i_score + ec_score) * conn_score
            
        # 4. 云计算得分
        # 需要考虑V2I、I2I、I2C通信队列和云处理队列
        # 使用平均V2I和I2I状态
        avg_v2i_score = np.mean([1 / (1 + x) for x in queue_states['v2i']])
        avg_i2i_score = np.mean([1 / (1 + x) for x in queue_states['i2i']])
        i2c_score = 1 / (1 + queue_states['i2c'])
        cc_score = 1 / (1 + queue_states['cc'])
        scores[-1] = (avg_v2i_score + avg_i2i_score + i2c_score + cc_score) / 4
        
        return scores