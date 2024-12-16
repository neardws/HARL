import gym
import copy
import numpy as np
from typing import List, Dict
from harl.common.valuenorm import ValueNorm
from utilities.noma import obtain_channel_gains_between_client_vehicle_and_server_vehicles, obtain_channel_gains_between_vehicles_and_edge_nodes
from utilities.wired_bandwidth import get_wired_bandwidth_between_edge_node_and_other_edge_nodes
from objects.task_object import task
from objects.vehicle_object import vehicle
from objects.edge_node_object import edge_node
from objects.cloud_server_object import cloud_server
from utilities.object_generation import generate_task_set, generate_vehicles, generate_edge_nodes, generate_cloud
from utilities.vehicle_classification import get_client_and_server_vehicles
from utilities.distance_and_coverage import get_distance_matrix_between_client_vehicles_and_server_vehicles, get_distance_matrix_between_vehicles_and_edge_nodes, get_distance_matrix_between_edge_nodes  
from utilities.distance_and_coverage import get_vehicles_under_V2I_communication_range, get_vehicles_under_V2V_communication_range
from queues.lc_queue import LCQueue
from queues.v2v_queue import V2VQueue
from queues.vc_queue import VCQueue
from queues.v2i_queue import V2IQueue
from queues.i2i_queue import I2IQueue
from queues.ec_queue import ECQueue
from queues.i2c_queue import I2CQueue
from queues.cc_queue import CCQueue
from lyapunov_optimization.virtual_queues.delay_queue import delayQueue
from lyapunov_optimization.virtual_queues.local_computing_resource_queue import lc_ressource_queue
from lyapunov_optimization.virtual_queues.vehicle_computing_resource_queue import vc_ressource_queue
from lyapunov_optimization.virtual_queues.edge_computing_resource_queue import ec_ressource_queue
from lyapunov_optimization.virtual_queues.cloud_compjuting_resource_queue import cc_ressource_queue

class VECEnv:
    def __init__(self, args):
        
        self.args = copy.deepcopy(args)
        
        # TODO: Add the values from the args dictionary

        self._slot_length: int = self.args["slot_length"]
        self._task_num: int = self.args["task_num"]
        self._task_distribution: str = self.args["task_distribution"]
        self._min_input_data_size_of_tasks: float = self.args["min_input_data_size_of_tasks"]
        self._max_input_data_size_of_tasks: float = self.args["max_input_data_size_of_tasks"]
        self._min_cqu_cycles_of_tasks: float = self.args["min_cqu_cycles_of_tasks"]
        self._max_cqu_cycles_of_tasks: float = self.args["max_cqu_cycles_of_tasks"]
        self._min_deadline_of_tasks: float = self.args["min_deadline_of_tasks"]
        self._max_deadline_of_tasks: float = self.args["max_deadline_of_tasks"]
        self._vehicle_num: int = self.args["vehicle_num"]
        self._vehicle_mobility_file_name_key: str = self.args["vehicle_mobility_file_name_key"]
        self._vehicular_trajectories_processing_start_time : str = self.args["vehicular_trajectories_processing_start_time"]
        self._vehicular_trajectories_processing_selection_way : str = self.args["vehicular_trajectories_processing_selection_way"]
        self._vehicular_trajectories_processing_filling_way : str = self.args["vehicular_trajectories_processing_filling_way"]
        self._vehicular_trajectories_processing_chunk_size : int = self.args["vehicular_trajectories_processing_chunk_size"]
        self._min_computing_capability_of_vehicles: float = self.args["min_computing_capability_of_vehicles"]
        self._max_computing_capability_of_vehicles: float = self.args["max_computing_capability_of_vehicles"]
        self._min_storage_capability_of_vehicles: float = self.args["min_storage_capability_of_vehicles"]
        self._max_storage_capability_of_vehicles: float = self.args["max_storage_capability_of_vehicles"]
        self._min_transmission_power_of_vehicles: float = self.args["min_transmission_power_of_vehicles"]
        self._max_transmission_power_of_vehicles: float = self.args["max_transmission_power_of_vehicles"]
        self._V2V_distance: float = self.args["V2V_distance"]
        self._min_task_arrival_rate_of_vehicles: float = self.args["min_task_arrival_rate_of_vehicles"]
        self._max_task_arrival_rate_of_vehicles: float = self.args["max_task_arrival_rate_of_vehicles"]
        self._vehicle_distribution: str = self.args["vehicle_distribution"]
        self._edge_num: int = self.args["edge_num"]
        self._min_computing_capability_of_edges: float = self.args["min_computing_capability_of_edges"]
        self._max_computing_capability_of_edges: float = self.args["max_computing_capability_of_edges"]
        self._min_storage_capability_of_edges: float = self.args["min_storage_capability_of_edges"]
        self._max_storage_capability_of_edges: float = self.args["max_storage_capability_of_edges"]
        self._min_communication_range_of_edges: float = self.args["min_communication_range_of_edges"]
        self._max_communication_range_of_edges: float = self.args["max_communication_range_of_edges"]
        self._I2I_transmission_rate: float = self.args["I2I_transmission_rate"]
        self._I2I_transmission_weight: float = self.args["I2I_transmission_weight"]
        self._edge_distribution: str = self.args["edge_distribution"]
        self._cloud_computing_capability: float = self.args["cloud_computing_capability"]
        self._cloud_storage_capability: float = self.args["cloud_storage_capability"]
        self._min_I2C_wired_bandwidth: float = self.args["min_I2C_wired_bandwidth"]
        self._max_I2C_wired_bandwidth: float = self.args["max_I2C_wired_bandwidth"]
        self._cloud_distribution: str = self.args["cloud_distribution"]
        self._V2V_bandwidth: float = self.args["V2V_bandwidth"]
        self._V2I_bandwidth: float = self.args["V2I_bandwidth"]
        self._white_gaussian_noise: int = self.args["white_gaussian_noise"]
        self._path_loss_exponent: int = self.args["path_loss_exponent"]
        self._task_seeds: list = self.args["task_seeds"]
        self._vehicle_seeds: list = self.args["vehicle_seeds"]
        self._edge_seeds: list = self.args["edge_seeds"]
        self._cloud_seed: int = self.args["cloud_seed"]
        
        # TODO: update the below values into the args dictionary
        self._maximum_client_vehicle_number = 10
        self._maximum_server_vehicle_number = 10
        self._edge_node_number = 10
        self._maximum_task_generation_number = 10
        
        self._maximum_task_offloaded_at_client_vehicle_number = 10
        self._maximum_task_offloaded_at_server_vehicle_number = 10
        self._maximum_task_offloaded_at_edge_node_number = 10
        self._maximum_task_offloaded_at_cloud_number = 10
        
        self.n_agents = ...
        self.state_space = self.generate_state_space()
        self.share_observation_space = self.repeat(self.state_space)
        self.observation_space = self.generate_observation_space()
        self.action_space = self.generate_action_space()
        
        self.value_normalizer = ValueNorm(1, device=self.device)
        
        self._task_offloaded_at_client_vehicles = {}
        self._task_offloaded_at_server_vehicles = {}
        self._task_offloaded_at_edge_nodes = {}
        self._task_offloaded_at_cloud = {}
        
        
        self._tasks : List[task] = generate_task_set(
            task_num=self._task_num,
            task_seeds=self._task_seeds,
            distribution=self._task_distribution,
            min_input_data_size=self._min_input_data_size_of_tasks,
            max_input_data_size=self._max_input_data_size_of_tasks,
            min_cqu_cycles=self._min_cqu_cycles_of_tasks,
            max_cqu_cycles=self._max_cqu_cycles_of_tasks,
            min_deadline=self._min_deadline_of_tasks,
            max_deadline=self._max_deadline_of_tasks,
        )
        
        # for task in self._tasks:
        #     print("task:\n", task)
        
        min_map_x, max_map_x, min_map_y, max_map_y, self._vehicles = generate_vehicles(
            vehicle_num=self._vehicle_num,
            vehicle_seeds=self._vehicle_seeds,
            slot_length=self._slot_length,
            file_name_key=self._vehicle_mobility_file_name_key,
            slection_way=self._vehicular_trajectories_processing_selection_way,
            filling_way=self._vehicular_trajectories_processing_filling_way,
            chunk_size=self._vehicular_trajectories_processing_chunk_size,
            start_time=self._vehicular_trajectories_processing_start_time,
            min_computing_capability=self._min_computing_capability_of_vehicles,
            max_computing_capability=self._max_computing_capability_of_vehicles,
            min_storage_capability=self._min_storage_capability_of_vehicles,
            max_storage_capability=self._max_storage_capability_of_vehicles,
            min_transmission_power=self._min_transmission_power_of_vehicles,
            max_transmission_power=self._max_transmission_power_of_vehicles,
            communication_range=self._V2V_distance,
            min_task_arrival_rate=self._min_task_arrival_rate_of_vehicles,
            max_task_arrival_rate=self._max_task_arrival_rate_of_vehicles,
            task_num=self._task_num,
            distribution=self._vehicle_distribution,
        )
        
        self._edge_nodes : List[edge_node] = generate_edge_nodes(
            edge_num=self._edge_num,
            edge_seeds=self._edge_seeds,
            min_map_x=min_map_x,
            max_map_x=max_map_x,
            min_map_y=min_map_y,
            max_map_y=max_map_y,
            min_computing_capability=self._min_computing_capability_of_edges,
            max_computing_capability=self._max_computing_capability_of_edges,
            min_storage_capability=self._min_storage_capability_of_edges,
            max_storage_capability=self._max_storage_capability_of_edges,
            communication_range=self._V2V_distance,
            distribution=self._edge_distribution,
            time_slot_num=self._slot_length,
        )
        
        self._cloud : cloud_server = generate_cloud(
            cloud_seed=self._cloud_seed,
            computing_capability=self._cloud_computing_capability,
            storage_capability=self._cloud_storage_capability,
            min_wired_bandwidth=self._min_I2C_wired_bandwidth,
            max_wired_bandwidth=self._max_I2C_wired_bandwidth,
            distribution=self._cloud_distribution,
            edge_node_num=self._edge_node_number,
            time_slot_num=self._slot_length,
        )
        
        self._distance_matrix_between_edge_nodes = get_distance_matrix_between_edge_nodes(
            edge_nodes=self._edge_nodes,
        )
        self._wired_bandwidths_between_edge_node_and_other_edge_nodes = get_wired_bandwidth_between_edge_node_and_other_edge_nodes(
            edge_nodes=self._edge_nodes,
            weight=profile.get_I2I_transmission_weight(),
            transmission_rate=profile.get_I2I_transmission_rate(),
            distance_matrix=self._distance_matrix_between_edge_nodes,
        )
        # print("distance_matrix_between_edge_nodes:\n", self._distance_matrix_between_edge_nodes)
        # print("wired_bandwidths_between_edge_node_and_other_edge_nodes:\n", self._wired_bandwidths_between_edge_node_and_other_edge_nodes)
        
        self._client_vehicles : List[vehicle] = []
        self._server_vehicles : List[vehicle] = []
        
        self._client_vehicles, self._server_vehicles = get_client_and_server_vehicles(
            now=self._now,
            vehicles=self._vehicles,
        )
        
        self._client_vehicle_num = len(self._client_vehicles)
        self._server_vehicle_num = len(self._server_vehicles)

        self._distance_matrix_between_client_vehicles_and_server_vehicles : np.ndarray = get_distance_matrix_between_client_vehicles_and_server_vehicles(
            client_vehicles=self._client_vehicles,
            server_vehicles=self._server_vehicles,
            now=self._now,
        )
        
        # print("self._distance_matrix_between_client_vehicles_and_server_vehicles:\n", self._distance_matrix_between_client_vehicles_and_server_vehicles)
        
        self._distance_matrix_between_client_vehicles_and_edge_nodes : np.ndarray = get_distance_matrix_between_vehicles_and_edge_nodes(
            client_vehicles=self._client_vehicles,
            edge_nodes=self._edge_nodes,
            now=self._now,
        )
        
        self._distance_matrix_between_edge_nodes : np.ndarray = get_distance_matrix_between_edge_nodes(
            edge_nodes=self._edge_nodes,
        )
        
        self._distance_matrix_between_edge_nodes_and_the_cloud : np.ndarray = get_distance_matrix_between_edge_nodes_and_the_cloud(
            edge_nodes=self._edge_nodes,
            cloud=self._cloud,
        )
        
        # print("self._distance_matrix_between_client_vehicles_and_edge_nodes:\n", self._distance_matrix_between_client_vehicles_and_edge_nodes)
        
        self._vehicles_under_V2V_communication_range : np.ndarray = get_vehicles_under_V2V_communication_range(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_server_vehicles,
            client_vehicles=self._client_vehicles,
            server_vehicles=self._server_vehicles,
        )
        
        # print("self._vehicles_under_V2V_communication_range:\n", self._vehicles_under_V2V_communication_range)
        
        self._vehicles_under_V2I_communication_range : np.ndarray = get_vehicles_under_V2I_communication_range(
            client_vehicles=self._client_vehicles,
            edge_nodes=self._edge_nodes,
            now=self._now,
        )
        
        # print("self._vehicles_under_V2I_communication_range:\n", self._vehicles_under_V2I_communication_range)
        
        self._channel_gains_between_client_vehicle_and_server_vehicles = obtain_channel_gains_between_client_vehicle_and_server_vehicles(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_server_vehicles,
            client_vehicles=self._client_vehicles,
            server_vehicles=self._server_vehicles,
            path_loss_exponent=self._env_profile.get_path_loss_exponent(),
        )
        
        # print("self._channel_gains_between_client_vehicle_and_server_vehicles:\n", self._channel_gains_between_client_vehicle_and_server_vehicles)
        
        self._channel_gains_between_client_vehicle_and_edge_nodes = obtain_channel_gains_between_vehicles_and_edge_nodes(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_edge_nodes,
            client_vehicles=self._client_vehicles,
            edge_nodes=self._edge_nodes,
            path_loss_exponent=self._env_profile.get_path_loss_exponent(),
        )
        
        # init the actual queues
        self._lc_queues = [LCQueue(
                time_slot_num=self._slot_length,
                name="lc_queue_" + str(_),
                client_vehicle_index=_,
            ) for _ in range(self._client_vehicle_num)]
        
        # TODO: need to be updated with the tasks
        self._v2v_queues = [V2VQueue(
                time_slot_num=self._slot_length,
                name="v2v_queue_" + str(_),
                server_vehicle_index=_,
                channel_gains_between_client_vehicle_and_server_vehicles=self._channel_gains_between_client_vehicle_and_server_vehicles,
            ) for _ in range(self._server_vehicle_num)]
        
        self._vc_queues = [VCQueue(
                time_slot_num=self._slot_length,
                name="vc_queue_" + str(_),
                server_vehicle_index=_,
            ) for _ in range(self._server_vehicle_num)]
        
        # TODO: need to be updated with the tasks
        self._v2i_queues = [V2IQueue(
                time_slot_num=self._slot_length,
                name="v2i_queue_" + str(_),
                edge_node_index=_,
                channel_gains_between_client_vehicle_and_edge_nodes=self._channel_gains_between_client_vehicle_and_edge_nodes,
            ) for _ in range(self._edge_node_number)]        
        
        # TODO need to be updated with the tasks
        self._i2i_queues = [I2IQueue(
                time_slot_num=self._slot_length,
                name="i2i_queue_" + str(_),
                edge_node_index=_,
                distance_matrix_between_edge_nodes=self._distance_matrix_between_edge_nodes,
                wired_bandwidths_between_edge_node_and_other_edge_nodes=self._wired_bandwidths_between_edge_node_and_other_edge_nodes,
            ) for _ in range(self._edge_node_number)]
        
        self._ec_queues = [ECQueue(
                time_slot_num=self._slot_length,
                name="ec_queue_" + str(_),
                edge_node_index=_,
            ) for _ in range(self._edge_node_number)]
        
        # TODO need to be updated with the tasks
        self._i2c_quque = I2CQueue(
                time_slot_num=self._slot_length,
                name="i2c_queue_" + str(_),
            )
        
        self._cc_queue = CCQueue(
                time_slot_num=self._slot_length,
                name="cc_queue_" + str(_),
            )
        
        # init the list to store the backlogs
        self._lc_queue_backlogs = np.zeros((self._client_vehicle_num, self._slot_length))
        self._v2v_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._vc_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._v2i_queue_backlogs = np.zeros((self._edge_node_number, self._slot_length))
        self._i2i_queue_backlogs = np.zeros((self._edge_node_number, self._slot_length))
        self._ec_queue_backlogs = np.zeros((self._edge_node_number, self._slot_length))
        self._i2c_queue_backlogs = np.zeros((1, self._slot_length))
        self._cc_queue_backlogs = np.zeros((1, self._slot_length))
        
        # init the virtual queues
        self._delay_queues = [delayQueue(
            time_slot_num=self._slot_length,
            name="delay_queue_" + str(_),
            task_index=_,
            client_vehicle_number=self._client_vehicle_num,
            maximum_task_generation_number=self._maximum_task_generation_number,
        ) for _ in range(self._task_num)]
        
        self._local_computing_resource_queues = [lc_ressource_queue(
            time_slot_num=self._slot_length,
            name="lc_ressource_queue_" + str(_),
            client_vehicle_index=_,
            maximum_task_generation_number=self._maximum_task_generation_number,
        ) for _ in range(self._client_vehicle_num)]
        
        self._vehicle_computing_resource_queues = [vc_ressource_queue(
            time_slot_num=self._slot_length,
            name="vc_ressource_queue_" + str(_),
            server_vehicle_index=_,
            server_vehicle_computing_capability=self._server_vehicles[_].get_computing_capability(),
        ) for _ in range(self._client_vehicle_num)]
            
        self._edge_computing_resource_queues = [ec_ressource_queue(
            time_slot_num=self._slot_length,
            name="ec_ressource_queue_" + str(_),
            edge_node_index=_,
            edge_node_computing_capability=self._edge_nodes[_].get_computing_capability(),
        ) for _ in range(self._edge_node_number)]
        
        self._cloud_computing_resource_queue = cc_ressource_queue(
            time_slot_num=self._slot_length,
            name="cc_ressource_queue_" + str(_),
            cloud_computing_capability=self._cloud.get_computing_capability(),
        )
        
        

    def step(self, actions):
        # transform the actions into the task offloading actions, transmission power allocation actions, computation resource allocation actions
        task_offloading_actions, transmission_power_allocation_actions, \
            computation_resource_allocation_actions = self.transform_actions(actions)
        
        self._task_offloaded_at_client_vehicles, self._task_offloaded_at_server_vehicles, \
            self._task_offloaded_at_edge_nodes, self._task_offloaded_at_cloud = self.obtain_tasks_offloading_conditions(
                task_offloading_actions=task_offloading_actions,
                vehicles_under_V2V_communication_range=self._vehicles_under_V2V_communication_range,
                vehicles_under_V2I_communication_range=self._vehicles_under_V2I_communication_range,
                now = self.cur_step,
            )
        
        # update the environment
        reward = self.compute_reward(
            task_offloading_actions=task_offloading_actions,
            transmission_power_allocation_actions=transmission_power_allocation_actions,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        
        self.cur_step += 1
        dones = [False for _ in range(self.n_agents)]
        if self.cur_step == self.max_cycles:
            dones = [True for _ in range(self.n_agents)]
        info = [{} for _ in range(self.n_agents)]
        obs = self.obtain_observation()
        s_obs = self.repeat(self.state())
        rewards = [[reward]] * self.n_agents
        return (
            obs,
            s_obs,
            rewards,
            dones,
            info,
            self.get_avail_actions(),
        )
        # return obs, state, rewards, dones, info, available_actions

    def reset(self):
        self._seed += 1
        self.cur_step = 0
        
        # TODO reset the environment
        
        obs = self.obtain_observation()
        s_obs = self.repeat(self.state())
        
        return obs, s_obs, self.get_avail_actions()
    
    def compute_reward(
        self,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        computation_resource_allocation_actions: Dict,
    ):
        # TODO implement the below method
        pass
    
    
    def obtain_observation(self):
        observations = []

        for agent_id in range(self.n_agents):
            if agent_id < self._maximum_client_vehicle_number:
                # 客户端车辆的观察
                observation = np.zeros(self.observation_space[agent_id].shape)
                index = 0
                
                # 任务信息
                tasks_of_vehicle = self._client_vehicles[agent_id].get_tasks_by_time(now=self.cur_step)
                min_num = min(len(tasks_of_vehicle), self._maximum_task_generation_number)
                for task in range(min_num):
                    task_data_size = tasks_of_vehicle[task][2].get_input_data_size()
                    cqu_cycles = tasks_of_vehicle[task][2].get_requested_computing_cycles()
                    observation[index] = task_data_size
                    index += 1
                    observation[index] = cqu_cycles
                    index += 1
                # 填充剩余的任务信息为零
                for _ in range(min_num, self._maximum_task_generation_number):
                    observation[index] = 0
                    index += 1
                    observation[index] = 0
                    index += 1

                # 客户端队列累积
                lc_backlog = self._lc_queue_backlogs[agent_id][self.cur_step]
                observation[index] = lc_backlog
                index += 1

                # V2V 连接信息
                for server in range(self._maximum_server_vehicle_number):
                    v2v_connection = self._vehicles_under_V2V_communication_range[agent_id][server]
                    observation[index] = v2v_connection
                    index += 1
                    
                # 服务车辆的队列累积
                for server in range(self._maximum_server_vehicle_number):
                    v2v_backlog = self._v2v_queue_backlogs[server][self.cur_step]
                    observation[index] = v2v_backlog
                    index += 1
                    vc_backlog = self._vc_queue_backlogs[server][self.cur_step]
                    observation[index] = vc_backlog
                    index += 1
                
                # V2I 连接信息
                for edge in range(self._edge_node_number):
                    v2i_connection = self._vehicles_under_V2I_communication_range[agent_id][edge]
                    observation[index] = v2i_connection
                    index += 1
                
                # 边缘节点的队列累积
                for _ in range(self._edge_node_number):
                    v2i_backlog = self._v2i_queue_backlogs[edge][self.cur_step]
                    observation[index] = v2i_backlog
                    index += 1
                    i2i_backlog = self._i2i_queue_backlogs[edge][self.cur_step]
                    observation[index] = i2i_backlog
                    index += 1
                    ec_backlog = self._ec_queue_backlogs[edge][self.cur_step]
                    observation[index] = ec_backlog
                    index += 1

                # 云的队列累积
                i2c_backlog = self._i2c_queue_backlogs[0][self.cur_step]
                observation[index] = i2c_backlog
                index += 1
                cc_backlog = self._cc_queue_backlogs[0][self.cur_step]
                observation[index] = cc_backlog
                
                observations.append(observation)

            elif agent_id < self._maximum_client_vehicle_number * 2:
                # 客户端车辆的另一种观察（功率分配）
                vehicle_index = agent_id - self._maximum_client_vehicle_number
                observation = np.zeros(self.observation_space[agent_id].shape)
                index = 0
                
                # 任务信息
                tasks_offloaded_at_vehicle = self._task_offloaded_at_client_vehicles["client_vehicle_" + str(vehicle_index)]
                min_mum = min(len(tasks_offloaded_at_vehicle), self._maximum_task_offloaded_at_client_vehicle_number)

                for task in range(min_num):
                    task_data_size = tasks_of_vehicle[task]["task"].get_input_data_size()
                    cqu_cycles = tasks_of_vehicle[task]["task"].get_requested_computing_cycles()
                    observation[index] = task_data_size
                    index += 1
                    observation[index] = cqu_cycles
                    index += 1
                # 填充剩余的任务信息为零
                for _ in range(min_num, self._maximum_task_offloaded_at_client_vehicle_number):
                    observation[index] = 0
                    index += 1
                    observation[index] = 0
                    index += 1

                # 客户端队列累积
                lc_backlog = self._lc_queue_backlogs[agent_id][self.cur_step]
                observation[index] = lc_backlog
                index += 1

                # V2V 连接信息
                for server in range(self._maximum_server_vehicle_number):
                    v2v_connection = self._vehicles_under_V2V_communication_range[agent_id][server]
                    observation[index] = v2v_connection
                    index += 1
                    
                # 服务车辆的队列累积
                for server in range(self._maximum_server_vehicle_number):
                    v2v_backlog = self._v2v_queue_backlogs[server][self.cur_step]
                    observation[index] = v2v_backlog
                    index += 1
                
                # V2I 连接信息
                for edge in range(self._edge_node_number):
                    v2i_connection = self._vehicles_under_V2I_communication_range[agent_id][edge]
                    observation[index] = v2i_connection
                    index += 1
                
                # 边缘节点的队列累积
                for _ in range(self._edge_node_number):
                    v2i_backlog = self._v2i_queue_backlogs[edge][self.cur_step]
                    observation[index] = v2i_backlog
                    index += 1

                observations.append(observation)

            elif agent_id < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number:
                # 服务器车辆的观察
                observation = np.zeros(self.observation_space[agent_id].shape)
                index = 0
                
                server_vehicle_index = agent_id - self._maximum_client_vehicle_number * 2
                # 获取任务信息和队列累积
                server_tasks = self._task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)]
                min_num = min(len(server_tasks), self._maximum_task_offloaded_at_server_vehicle_number)
                for task in range(min_num):
                    task_data_size = server_tasks[task]["task"].get_input_data_size()
                    cqu_cycles = server_tasks[task]["task"].get_requested_computing_cycles()
                    observation[index] = task_data_size
                    index += 1
                    observation[index] = cqu_cycles
                    index += 1
                for _ in range(min_num, self._maximum_task_offloaded_at_server_vehicle_number):
                    observation[index] = 0
                    index += 1
                    observation[index] = 0
                    index += 1

                backlog = self._vc_queue_backlogs[agent_id - self._maximum_client_vehicle_number * 2][self.cur_step]
                observation[index] = backlog
                index += 1
                
                observations.append(observation)

            # 处理边缘节点和云的观察，与上述类似
            # TODO: @llf-cpu

        return observations

    
    def state(self):
        # 初始化一个状态数组，形状为 (state_space_number,)
        state = np.zeros(self.state_space.shape)  # self.state_space 是通过 generate_state_space 生成的

        index = 0

        # 1. 客户端车辆的任务信息
        for client in range(self._maximum_client_vehicle_number):
            tasks_of_vehicle = self._client_vehicles[client].get_tasks_by_time(now=self.cur_step)  # 准备获取任务信息
            min_num = min(len(tasks_of_vehicle), self._maximum_task_generation_number)
            for task in range(min_num):
                task_data_size = tasks_of_vehicle[task][2].get_input_data_size()
                cqu_cycles = tasks_of_vehicle[task][2].get_requested_computing_cycles()
                state[index] = task_data_size
                index += 1
                state[index] = cqu_cycles
                index += 1
            for _ in range(min_num, self._maximum_task_generation_number):
                state[index] = 0
                index += 1
                state[index] = 0
                index += 1

        # 2. 客户端车辆的队列累积
        for client in range(self._maximum_client_vehicle_number):
            lc_backlog = self._lc_queue_backlogs[client][self.cur_step]  # 获取客户端车辆的队列累积
            state[index] = lc_backlog
            index += 1

        # 3. V2V 连接信息
        for client in range(self._maximum_client_vehicle_number):
            for server in range(self._maximum_server_vehicle_number):
                v2v_connection = self._vehicles_under_V2V_communication_range[client][server]  # 获取 V2V 连接信息
                state[index] = v2v_connection
                index += 1
                
        # 4. 服务器车辆的队列累积
        for server in range(self._maximum_server_vehicle_number):
            v2v_backlog = self._v2v_queue_backlogs[server][self.cur_step]  # 获取服务器车辆的队列累积
            state[index] = v2v_backlog
            index += 1
            vc_backlog = self._vc_queue_backlogs[server][self.cur_step]  # 获取服务器车辆的队列累积
            state[index] = vc_backlog
            index += 1

        # 5. V2I 连接信息
        for client in range(self._maximum_client_vehicle_number):
            for edge in range(self._edge_node_number):
                v2i_connection = self._vehicles_under_V2I_communication_range[client][edge]
                state[index] = v2i_connection
                index += 1

        # 6. 边缘节点的队列累积
        for edge in range(self._edge_node_number):
            v2i_backlog = self._v2i_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            i2i_backlog = self._i2i_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            ec_backlog = self._ec_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            state[index] = v2i_backlog
            index += 1
            state[index] = i2i_backlog
            index += 1
            state[index] = ec_backlog
            index += 1

        # 7. 云的队列累积（I2C 和 CC）
        i2c_backlog = self._i2c_queue_backlogs[0][self.cur_step]  # 获取云的 I2C 队列
        state[index] = i2c_backlog
        index += 1
        cc_backlog = self._cc_queue_backlogs[0][self.cur_step]  # 获取云的 CC 队列
        state[index] = cc_backlog
        
        return state

        
    def seed(self, seed):
        self._seed = seed

    def render(self):
        raise NotImplementedError

    def close(self):
        pass
        
    def transform_actions(self, actions):
        task_offloading_actions = {}
        transmission_power_allocation_actions = {}
        computation_resource_allocation_actions = {}
        for i in range(self._n_agents):
            if i < self._maximum_client_vehicle_number:
                for j in range(self._maximum_task_generation_number):
                    if actions[i][j] == 0:
                        task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] = "Local"
                    elif actions[i][j] >= 1 and actions[i][j] <= 1 + self._maximum_server_vehicle_number:
                        task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] = "Server Vehicle " + str(actions[i][j] - 1)
                    elif actions[i][j] >= 1 + self._maximum_server_vehicle_number and actions[i][j] <= 1 + self._maximum_server_vehicle_number + self._edge_node_number:
                        task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] = "Edge Node " + str(actions[i][j] - 1 - self._maximum_server_vehicle_number)
                    else:
                        task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] = "Cloud"
            elif i < self._maximum_client_vehicle_number * 2:
                client_vehicle_number = i - self._maximum_client_vehicle_number
                computation_resource_allocation = actions[i][0:self._maximum_task_offloaded_at_client_vehicle_number]
                computation_resource_allocation_normalized = self.value_normalizer.normalize(computation_resource_allocation)
                transmission_power_allocation = actions[i][-2:]
                transmission_power_allocation_normalized = self.value_normalizer.normalize(transmission_power_allocation)
                # V2V transmission power allocation + V2I transmission power allocation
                transmission_power_allocation_actions["client_vehicle_" + str(client_vehicle_number)] = transmission_power_allocation_normalized
                computation_resource_allocation_actions["client_vehicle_" + str(client_vehicle_number)] = computation_resource_allocation_normalized
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number:
                server_vehicle_number = i - self._maximum_client_vehicle_number * 2
                computation_resource_allocation = actions[i]
                computation_resource_allocation_normalized = self.value_normalizer.normalize(computation_resource_allocation)
                computation_resource_allocation_actions["server_vehicle_" + str(server_vehicle_number)] = computation_resource_allocation_normalized
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number + self._edge_node_number:
                edge_node_number = i - self._maximum_client_vehicle_number * 2 - self._maximum_server_vehicle_number
                computation_resource_allocation = actions[i]
                computation_resource_allocation_normalized = self.value_normalizer.normalize(computation_resource_allocation)
                computation_resource_allocation_actions["edge_node_" + str(edge_node_number)] = computation_resource_allocation_normalized
            else:
                computation_resource_allocation = actions[i]
                computation_resource_allocation_normalized = self.value_normalizer.normalize(computation_resource_allocation)
                computation_resource_allocation_actions["cloud"] = computation_resource_allocation_normalized
        return task_offloading_actions, transmission_power_allocation_actions, computation_resource_allocation_actions
    
        
    def obtain_tasks_offloading_conditions(
        self, 
        task_offloading_actions,
        vehicles_under_V2V_communication_range,
        vehicles_under_V2I_communication_range,
        now: int,
    ):
        task_offloaded_at_client_vehicles = {}
        task_offloaded_at_server_vehicles = {}
        task_offloaded_at_edge_nodes = {}
        task_offloaded_at_cloud = {}
        
        # init 
        for i in range(self._maximum_client_vehicle_number):
            task_offloaded_at_client_vehicles["client_vehicle_" + str(i)] = []
        for i in range(self._maximum_server_vehicle_number):
            task_offloaded_at_server_vehicles["server_vehicle_" + str(i)] = []
        for i in range(self._edge_node_number):
            task_offloaded_at_edge_nodes["edge_node_" + str(i)] = []
        task_offloaded_at_cloud["cloud"] = []
        
        for i in range(self._maximum_client_vehicle_number):
            tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number)
            if min_num > 0:
                for j in range(min_num):
                    task_index = tasks_of_vehicle_i[j][1]
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Local":
                        if len(task_offloaded_at_client_vehicles["client_vehicle_" + str(i)]) < self._maximum_task_offloaded_at_client_vehicle_number:
                            task_offloaded_at_client_vehicles["client_vehicle_" + str(i)].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Server Vehicle"):
                        server_vehicle_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        if vehicles_under_V2V_communication_range[i][server_vehicle_index] == 1:
                            if len(task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)]) < self._maximum_task_offloaded_at_server_vehicle_number:
                                task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Edge Node"):
                        edge_node_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        if any(vehicles_under_V2I_communication_range[i][e] == 1 for e in range(self._edge_node_number)):
                            if len(task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)]) < self._maximum_task_offloaded_at_edge_node_number:
                                task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                    else:
                        if any(vehicles_under_V2I_communication_range[i][e] == 1 for e in range(self._edge_node_number)):
                            if len(task_offloaded_at_cloud["cloud"]) < self._maximum_task_offloaded_at_cloud_number:
                                task_offloaded_at_cloud["cloud"].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                                
        return task_offloaded_at_client_vehicles, task_offloaded_at_server_vehicles, task_offloaded_at_edge_nodes, task_offloaded_at_cloud
    
    
    def generate_action_space(self):
        if self.n_agents != self._maximum_client_vehicle_number * 2 + self._edge_node_number + 1:
            raise ValueError("The number of agents is not correct.")
        
        task_offloading_number = 1 + self._maximum_server_vehicle_number + self._maximum_server_vehicle_number + self._edge_node_number + 1
        
        action_space = []
        for i in range(self.n_agents):
            if i < self._maximum_client_vehicle_number:
                # task offloading decision
                action_space.append(gym.spaces.multi_discrete.MultiDiscrete([task_offloading_number] * self._maximum_task_generation_number))
            elif i < self._maximum_client_vehicle_number * 2:
                # transmission power allocation and computation resource allocation
                # need to be normalized, the sum of the elements in the first maximum_task_offloaded_at_client_vehicle_number elements should be 1
                # wihch means the computation resource allocation cannot exceed the capacity of the client vehicle
                # the sum of the elements in the last two elements should be 1
                # which means the sum of the transmission power allocation should be 1
                action_space.append(gym.spaces.Box(low=0, high=1, shape=(self._maximum_task_offloaded_at_client_vehicle_number + 2,), dtype=np.float32))
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number:
                # computation resource allocation
                # need to be normalized, the sum of the elements should be 1
                action_space.append(gym.spaces.Box(low=0, high=1, shape=(self._maximum_task_offloaded_at_server_vehicle_number,), dtype=np.float32))
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number + self._edge_node_number:
                # computation resource allocation
                # need to be normalized, the sum of the elements should be 1
                action_space.append(gym.spaces.Box(low=0, high=1, shape=(self._maximum_task_offloaded_at_edge_node_number,), dtype=np.float32))
            else:
                # computation resource allocation
                # need to be normalized, the sum of the elements should be 1
                action_space.append(gym.spaces.Box(low=0, high=1, shape=(self._maximum_task_offloaded_at_cloud_number,), dtype=np.float32))
                
        return action_space
    
    def generate_observation_space(self):
        if self.n_agents != self._maximum_client_vehicle_number * 2 + self._edge_node_number + 1:
            raise ValueError("The number of agents is not correct.")
        
        observation_space = []
        
        for i in range(self.n_agents):
            if i < self._maximum_client_vehicle_number:
                # the observation space of the client vehicle
                # conisits of task information (data size, CQU cycles)
                # the queue backlog of the lc_queue
                # the V2V connection based on the V2V distance
                # the queue backlog of the V2V and VC queue of all server clients
                # the V2I connection based on the V2I distance
                # the queue backlog of the V2I, I2I, EC queue of all edge nodes
                # the queue backlog of the I2C and CC queue of the cloud
                client_vehicle_observation = \
                    2 * self._maximum_task_generation_number + \
                    1 + \
                    self._maximum_server_vehicle_number + \
                    2 * self._maximum_server_vehicle_number + \
                    self._edge_node_number + \
                    3 * self._edge_node_number + \
                    2
                observation_space.append(gym.spaces.Box(low=0, high=1, shape=(client_vehicle_observation,), dtype=np.float32))
            elif i < self._maximum_client_vehicle_number * 2:
                # the observation space of the client vehcile when doing the transmission power allocation and computation resource allocation
                # consists of task information (data size, CQU cycles)
                # the queue backlog of the lc_queue
                # the V2V connection based on the V2V distance 
                # the queue backlog of the V2V queue of all server clients
                # the V2I connection based on the V2I distance
                # the queue backlog of the V2I queue of all edge nodes
                client_vehicle_observation_2 = \
                    2 * self._maximum_task_offloaded_at_client_vehicle_number + \
                    1 + \
                    self._maximum_server_vehicle_number + \
                    self._maximum_server_vehicle_number + \
                    self._edge_node_number + \
                    self._edge_node_number
                observation_space.append(gym.spaces.Box(low=0, high=1, shape=(client_vehicle_observation_2, ), dtype=np.float32))
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number:
                # the observation space of the server vehicle
                # consists of the task information (data size, CQU cycles)
                # the queue backlog of the VC queue
                server_vehicle_observation = \
                    2 * self._maximum_task_offloaded_at_server_vehicle_number + \
                    1
                observation_space.append(gym.spaces.Box(low=0, high=1, shape=(server_vehicle_observation, ), dtype=np.float32))
            
            elif i < self._maximum_client_vehicle_number * 2 + self._maximum_server_vehicle_number + self._edge_node_number:
                # the observation space of the edge node
                # consists of the task information (data size, CQU cycles)
                # the queue backlog of the EC queue
                edge_node_observation = \
                    2 * self._maximum_task_offloaded_at_edge_node_number + \
                    1
                observation_space.append(gym.spaces.Box(low=0, high=1, shape=(edge_node_observation, ), dtype=np.float32))
            else:
                # the observation space of the cloud
                # consists of the task information (data size, CQU cycles)
                # the queue backlog of the CC queue
                cloud_observation = \
                    2 * self._maximum_task_offloaded_at_cloud_number + \
                    1
                observation_space.append(gym.spaces.Box(low=0, high=1, shape=(cloud_observation, ), dtype=np.float32))
        return observation_space
    
    def generate_state_space(self):
        # the state space of the system
        # conisits of task information (data size, CQU cycles) of all client vehicles
        # the queue backlog of the lc_queue of all client vehicles
        # the V2V connection based on the V2V distance of all client vehicles
        # the queue backlog of the V2V and VC queue of all server clients of all server vehicles
        # the V2I connection based on the V2I distance of all client vehicles
        # the queue backlog of the V2I, I2I, EC queue of all edge nodes 
        # the queue backlog of the I2C and CC queue of the cloud
        state_space_number = \
            self._maximum_client_vehicle_number * 2 * self._maximum_task_generation_number + \
            self._maximum_client_vehicle_number + \
            self._maximum_client_vehicle_number * self._maximum_server_vehicle_number + \
            2 * self._maximum_server_vehicle_number + \
            self._maximum_client_vehicle_number * self._edge_node_number + \
            3 * self._edge_node_number + \
            2
        return gym.spaces.Box(low=0, high=1, shape=(state_space_number, ), dtype=np.float32)
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def get_avail_actions(self):
        return None
    