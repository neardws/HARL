import gym
import math
import copy
import numpy as np
import random
import time
from typing import List, Dict, Tuple
from utilities.noma import obtain_channel_gains_between_client_vehicle_and_server_vehicles, obtain_channel_gains_between_vehicles_and_edge_nodes
from utilities.wired_bandwidth import get_wired_bandwidth_between_edge_node_and_other_edge_nodes
from objects.vehicle_object import vehicle
from objects.edge_node_object import edge_node
from objects.cloud_server_object import cloud_server
from utilities.object_generation import generate_task_set, generate_vehicles, generate_edge_nodes, generate_cloud
from utilities.vehicle_classification import get_client_and_server_vehicles
from utilities.distance_and_coverage import get_distance_matrix_between_client_vehicles_and_server_vehicles, get_distance_matrix_between_vehicles_and_edge_nodes, get_distance_matrix_between_edge_nodes, get_distance_matrix_between_edge_nodes_and_the_cloud  
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
from lyapunov_optimization.lyapunov_drift_plus_penalty.cost import compute_lc_computing_cost, compute_vc_computing_cost, compute_ec_computing_cost, compute_cc_computing_cost, compute_v2v_transmission_cost, compute_v2i_transmission_cost, compute_i2i_transmission_cost, compute_i2c_transmission_cost, compute_total_cost
from sklearn.decomposition import PCA


class VECEnv:
    def __init__(self, args):
        
        self.args = copy.deepcopy(args)
        
        self._episode_index = 0
        self.cur_step = 0

        self._slot_length: int = self.args["slot_length"]
        
        self._task_num: int = self.args["task_num"]
        self._task_distribution: str = self.args["task_distribution"]
        self._min_input_data_size_of_tasks: float = self.args["min_input_data_size_of_tasks"]
        self._max_input_data_size_of_tasks: float = self.args["max_input_data_size_of_tasks"]
        self._min_cpu_cycles_of_tasks: float = self.args["min_cpu_cycles_of_tasks"]
        self._max_cpu_cycles_of_tasks: float = self.args["max_cpu_cycles_of_tasks"]
        
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
        self._server_vehicle_probability: float = self.args["server_vehicle_probability"]
        self._task_ids_rate: float = self.args["task_ids_rate"]
        
        self._edge_num: int = self.args["edge_num"]
        self._min_computing_capability_of_edges: float = self.args["min_computing_capability_of_edges"]
        self._max_computing_capability_of_edges: float = self.args["max_computing_capability_of_edges"]
        self._min_storage_capability_of_edges: float = self.args["min_storage_capability_of_edges"]
        self._max_storage_capability_of_edges: float = self.args["max_storage_capability_of_edges"]
        self._min_communication_range_of_edges: float = self.args["min_communication_range_of_edges"]
        self._max_communication_range_of_edges: float = self.args["max_communication_range_of_edges"]
        self._I2I_transmission_rate: float = self.args["I2I_transmission_rate"]
        self._I2I_transmission_weight: float = self.args["I2I_transmission_weight"]
        self._I2I_propagation_speed: float = self.args["I2I_propagation_speed"]
        self._edge_distribution: str = self.args["edge_distribution"]
        
        self._cloud_computing_capability: float = self.args["cloud_computing_capability"]
        self._cloud_storage_capability: float = self.args["cloud_storage_capability"]
        self._min_I2C_wired_bandwidth: float = self.args["min_I2C_wired_bandwidth"]
        self._max_I2C_wired_bandwidth: float = self.args["max_I2C_wired_bandwidth"]
        self._I2C_transmission_rate: float = self.args["I2C_transmission_rate"]
        self._I2C_propagation_speed: float = self.args["I2C_propagation_speed"]
        self._cloud_distribution: str = self.args["cloud_distribution"]
        
        self._V2V_bandwidth: float = self.args["V2V_bandwidth"]
        self._V2I_bandwidth: float = self.args["V2I_bandwidth"]
        self._white_gaussian_noise: int = self.args["white_gaussian_noise"]
        self._path_loss_exponent: int = self.args["path_loss_exponent"]

        self._v2v_n_components: int = self.args["v2v_n_components"]
        self._v2i_n_components: int = self.args["v2i_n_components"]  

        self._using_PPO = self.args["using_PPO"]
        self._algorithm_type = self.args["algorithm_type"]

        self.v2v_pca = PCA(n_components=self._v2v_n_components)  # 设定 V2V 压缩后的维度
        self.v2i_pca = PCA(n_components=self._v2i_n_components)  # 设定 V2I 压缩后的维度
        
        self._maximum_server_vehicle_num = 5
        
        self._seed = self.args["seed"]
        # generate the seeds for the tasks
        np.random.seed(self._seed)
        random.seed(self._seed)

        # Generate deterministic seeds using fixed offsets
        self._task_seeds = np.array([self._seed + i for i in range(self._task_num)])
        self._vehicle_seeds = np.array([self._seed + self._task_num + i for i in range(self._vehicle_num)])
        self._edge_seeds = np.array([self._seed + self._task_num + self._vehicle_num + i for i in range(self._edge_num)])
        self._cloud_seed = self._seed + self._task_num + self._vehicle_num + self._edge_num
        
        self._penalty_weight = self.args["penalty_weight"]

        self.min_reward = None
        self.max_reward = None
        self.reward_alpha = 0.03  # 移动平均系数
                        
        self._task_offloaded_at_client_vehicles = {}
        self._task_offloaded_at_server_vehicles = {}
        self._task_offloaded_at_edge_nodes = {}
        self._task_uploaded_at_edge_nodes = {}
        self._task_offloaded_at_cloud = {}
        
        self._tasks : List[Dict] = generate_task_set(
            task_num=self._task_num,
            task_seeds=self._task_seeds,
            distribution=self._task_distribution,
            min_input_data_size=self._min_input_data_size_of_tasks,
            max_input_data_size=self._max_input_data_size_of_tasks,
            min_cpu_cycles=self._min_cpu_cycles_of_tasks,
            max_cpu_cycles=self._max_cpu_cycles_of_tasks,
        )
        
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
            server_vehicle_probability=self._server_vehicle_probability,
            task_num=self._task_num,
            task_ids_rate=self._task_ids_rate,
            distribution=self._vehicle_distribution,
            tasks=self._tasks,
        )
        
        self._maximum_task_data_size, self._maximum_task_required_cycles = self.obtain_maximum_task_data_size_and_required_cycles_of_vehicles(
            vehicles=self._vehicles,
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
            min_map_x=min_map_x,
            max_map_x=max_map_x,
            min_map_y=min_map_y,
            max_map_y=max_map_y,
            computing_capability=self._cloud_computing_capability,
            storage_capability=self._cloud_storage_capability,
            min_wired_bandwidth=self._min_I2C_wired_bandwidth,
            max_wired_bandwidth=self._max_I2C_wired_bandwidth,
            distribution=self._cloud_distribution,
            edge_node_num=self._edge_num,
            time_slot_num=self._slot_length,
        )
        
        self._distance_matrix_between_edge_nodes = get_distance_matrix_between_edge_nodes(
            edge_nodes=self._edge_nodes,
        )
        
        self._wired_bandwidths_between_edge_node_and_other_edge_nodes = get_wired_bandwidth_between_edge_node_and_other_edge_nodes(
            edge_nodes=self._edge_nodes,
            weight=self._I2I_transmission_weight,
            transmission_rate=self._I2I_transmission_rate,
            distance_matrix=self._distance_matrix_between_edge_nodes,
        )
        
        self._client_vehicles : List[vehicle] = []
        self._server_vehicles : List[vehicle] = []
        
        self._client_vehicles, self._server_vehicles = get_client_and_server_vehicles(
            vehicles=self._vehicles,
        )
        
        # print("client_vehicles_number: ", len(self._client_vehicles))
        # print("server_vehicles_number: ", len(self._server_vehicles))
        
        self._maximum_task_generation_number_of_vehicles, self._average_task_generation_numbers, \
            self._sum_task_generation_numbers= self.obtain_task_generation_numbers(
            client_vehicles = self._client_vehicles,
        )
        
        # print("self._maximum_task_generation_number_of_vehicles: ", self._maximum_task_generation_number_of_vehicles)    
        # print("self._average_task_generation_numbers: ", self._average_task_generation_numbers)
        # print("self._sum_task_generation_numbers: ", self._sum_task_generation_numbers)
        
        
        self._maximum_task_offloaded_at_client_vehicle_number, self._maximum_task_offloaded_at_server_vehicle_number, \
            self._maximum_task_offloaded_at_edge_node_number, self._maximum_task_offloaded_at_cloud_number = self.obtain_offloading_numbers(
                average_task_generation_numbers=self._average_task_generation_numbers,
                sum_task_generation_numbers=self._sum_task_generation_numbers,
            )
        
        self._client_vehicle_num = len(self._client_vehicles)
        self._server_vehicle_num = len(self._server_vehicles)

        self._distance_matrix_between_client_vehicles_and_server_vehicles : np.ndarray = get_distance_matrix_between_client_vehicles_and_server_vehicles(
            client_vehicles=self._client_vehicles,
            server_vehicles=self._server_vehicles,
            time_slot_num=self._slot_length,
        )
                
        self._distance_matrix_between_client_vehicles_and_edge_nodes : np.ndarray = get_distance_matrix_between_vehicles_and_edge_nodes(
            client_vehicles=self._client_vehicles,
            edge_nodes=self._edge_nodes,
            time_slot_num=self._slot_length,
        )
        
        self._vehicles_under_V2V_communication_range : np.ndarray = get_vehicles_under_V2V_communication_range(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_server_vehicles,
            client_vehicles=self._client_vehicles,
            server_vehicles=self._server_vehicles,
            time_slot_num=self._slot_length,
        )

        # 收集初始训练数据
        initial_v2v_data = self.collect_initial_v2v_data()

        # 拟合 PCA 模型
        self.v2v_pca.fit(initial_v2v_data)


        # 收集用于拟合 V2I PCA 的初始数据
        initial_v2i_data = self.collect_initial_v2i_data()

        # 拟合 PCA 模型
        self.v2i_pca.fit(initial_v2i_data)
        
        self._vehicles_under_V2I_communication_range : np.ndarray = get_vehicles_under_V2I_communication_range(
            client_vehicles=self._client_vehicles,
            edge_nodes=self._edge_nodes,
            time_slot_num=self._slot_length,
        )
        
        self._channel_gains_between_client_vehicle_and_server_vehicles = obtain_channel_gains_between_client_vehicle_and_server_vehicles(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_server_vehicles,
            client_vehicles=self._client_vehicles,
            time_slot_num=self._slot_length,
            server_vehicles=self._server_vehicles,
            path_loss_exponent=self._path_loss_exponent,
        )
    
        self._channel_gains_between_client_vehicle_and_edge_nodes = obtain_channel_gains_between_vehicles_and_edge_nodes(
            distance_matrix=self._distance_matrix_between_client_vehicles_and_edge_nodes,
            client_vehicles=self._client_vehicles,
            time_slot_num=self._slot_length,
            edge_nodes=self._edge_nodes,
            path_loss_exponent=self._path_loss_exponent,
        )
        
        self._distance_matrix_between_edge_nodes_and_the_cloud : np.ndarray = get_distance_matrix_between_edge_nodes_and_the_cloud(
            edge_nodes=self._edge_nodes,
            cloud_server=self._cloud,
        )
        
        # init the maximum costs
        self._maximum_v2v_transmission_costs = np.zeros((self._client_vehicle_num, ))
        self._maximum_v2i_transmission_costs = np.zeros((self._client_vehicle_num, ))
        self._maximum_lc_computing_costs = np.zeros((self._client_vehicle_num, ))
        
        self._maximum_vc_computing_costs = np.zeros((self._server_vehicle_num, ))
        
        self._maximum_ec_computing_costs = np.zeros((self._edge_num, ))
        self._maximum_i2i_transmission_costs = np.zeros((self._edge_num, ))
        self._maximum_i2c_transmission_costs = np.zeros((self._edge_num, ))
        
        self._maximum_cc_computing_cost = 0.0
        
        # init the maximum phi t
        self._maximum_phi_t_delay_queue = np.zeros((self._client_vehicle_num, ))
        self._maximum_phi_t_lc_queue = np.zeros((self._client_vehicle_num, ))
        self._maximum_phi_t_vc_queue = np.zeros((self._server_vehicle_num, ))
        self._maximum_phi_t_ec_queue = np.zeros((self._edge_num, ))
        self._maximum_phi_t_cc_queue = 0.0
        
        self._minimum_phi_t_delay_queue = np.full((self._client_vehicle_num,), np.inf)
        self._minimum_phi_t_lc_queue = np.full((self._client_vehicle_num,), np.inf)
        self._minimum_phi_t_vc_queue = np.full((self._server_vehicle_num,), np.inf)
        self._minimum_phi_t_ec_queue = np.full((self._edge_num,), np.inf)
        self._minimum_phi_t_cc_queue = np.inf
        
        self.n_agents : int = 2
        
        self.state_space = self.generate_state_space()
        self.share_observation_space = self.repeat(self.state_space)
        
        self._max_observation, self._task_offloading_agent_observation_number, \
            self._resource_allocation_agent_observation_number  = self.init_observation_number_of_agents()
        
        self.observation_space = self.generate_observation_space()
        self.true_observation_space = self.generate_true_observation_space()
        
        self._max_action, self._task_offloading_agent_action_number, \
            self._resource_allocation_agent_action_number = self.init_action_number_of_agents()
        
        self.action_space = self.generate_action_space()
        self.true_action_space = self.generate_true_action_space()
        
        # init the actual queues
        self._lc_queues = [LCQueue(
            time_slot_num=self._slot_length,
            name="lc_queue_" + str(_),
            client_vehicle_index=_,
            maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
        ) for _ in range(self._client_vehicle_num)]
        
        self._v2v_queues = [V2VQueue(
                time_slot_num=self._slot_length,
                name="v2v_queue_" + str(_),
                server_vehicle_index=_,
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                white_gaussian_noise=self._white_gaussian_noise,
                V2V_bandwidth=self._V2V_bandwidth,
                channel_gains_between_client_vehicle_and_server_vehicles=self._channel_gains_between_client_vehicle_and_server_vehicles,
            ) for _ in range(self._server_vehicle_num)]
        
        # print("self._client_vehicle_num: ", self._client_vehicle_num)
        # print("self._server_vehicle_num: ", self._server_vehicle_num)
        
        # print("v2v_queues: ", self._v2v_queues)
        
        self._vc_queues = [VCQueue(
                time_slot_num=self._slot_length,
                name="vc_queue_" + str(_),
                server_vehicle_index=_,
            ) for _ in range(self._server_vehicle_num)]

        # print("vc_queues: ", self._vc_queues)
        
        # print("* " * 50)
        
        self._v2i_queues = [V2IQueue(
                time_slot_num=self._slot_length,
                name="v2i_queue_" + str(_),
                edge_node_index=_,
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                white_gaussian_noise=self._white_gaussian_noise,
                V2I_bandwidth=self._V2I_bandwidth, 
                channel_gains_between_client_vehicle_and_edge_nodes=self._channel_gains_between_client_vehicle_and_edge_nodes,
            ) for _ in range(self._edge_num)]        
        
        self._i2i_queues = [I2IQueue(
                time_slot_num=self._slot_length,
                name="i2i_queue_" + str(_),
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                edge_node_number=self._edge_num,
                edge_node_index=_,
                white_gaussian_noise=self._white_gaussian_noise,
                V2I_bandwidth=self._V2I_bandwidth,
                I2I_transmission_rate=self._I2I_transmission_rate,
                I2I_propagation_speed=self._I2I_propagation_speed,
                channel_gains_between_client_vehicle_and_edge_nodes=self._channel_gains_between_client_vehicle_and_edge_nodes,
            ) for _ in range(self._edge_num)]
        
        self._ec_queues = [ECQueue(
                time_slot_num=self._slot_length,
                name="ec_queue_" + str(_),
                edge_node_index=_,
            ) for _ in range(self._edge_num)]
        
        self._i2c_quque = I2CQueue(
                time_slot_num=self._slot_length,
                name="i2c_queue",
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                edge_node_number=self._edge_num,
                white_gaussian_noise=self._white_gaussian_noise,
                V2I_bandwidth=self._V2I_bandwidth,
                I2C_transmission_rate=self._I2C_transmission_rate,
                I2C_propagation_speed=self._I2C_propagation_speed,
                channel_gains_between_client_vehicle_and_edge_nodes=self._channel_gains_between_client_vehicle_and_edge_nodes,
            )
        
        self._cc_queue = CCQueue(
                time_slot_num=self._slot_length,
                name="cc_queue",
            )
        
        # init the list to store the backlogs
        self._lc_queue_backlogs = np.zeros((self._client_vehicle_num, self._slot_length))
        self._v2v_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._vc_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._v2i_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._i2i_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._ec_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._i2c_queue_backlogs = np.zeros((self._slot_length, ))
        self._cc_queue_backlogs = np.zeros((self._slot_length, ))
        
        # init the maximum queue length
        self._maximum_lc_queue_length = np.zeros((self._client_vehicle_num, ))
        self._maximum_v2v_queue_length = np.zeros((self._server_vehicle_num, ))
        self._maximum_vc_queue_length = np.zeros((self._server_vehicle_num, ))
        self._maximum_v2i_queue_length = np.zeros((self._edge_num, ))
        self._maximum_i2i_queue_length = np.zeros((self._edge_num, ))
        self._maximum_ec_queue_length = np.zeros((self._edge_num, ))
        self._maximum_i2c_queue_length = 0.0
        self._maximum_cc_queue_length = 0.0
        
        # init the virtual queues
        self._delay_queues = [delayQueue(
            time_slot_num=self._slot_length,
            name="delay_queue_" + str(_),
            client_vehicle_index=_,
            client_vehicle_number=self._client_vehicle_num,
            maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
        ) for _ in range(self._client_vehicle_num)]
        
        self._local_computing_resource_queues = [lc_ressource_queue(
            time_slot_num=self._slot_length,
            name="lc_ressource_queue_" + str(_),
            client_vehicle_index=_,
            client_vehicle_computing_capability=self._client_vehicles[_].get_computing_capability(),
        ) for _ in range(self._client_vehicle_num)]
        
        self._vehicle_computing_resource_queues = [vc_ressource_queue(
            time_slot_num=self._slot_length,
            name="vc_ressource_queue_" + str(_),
            server_vehicle_index=_,
            server_vehicle_computing_capability=self._server_vehicles[_].get_computing_capability(),
        ) for _ in range(self._server_vehicle_num)]
            
        self._edge_computing_resource_queues = [ec_ressource_queue(
            time_slot_num=self._slot_length,
            name="ec_ressource_queue_" + str(_),
            edge_node_index=_,
            edge_node_computing_capability=self._edge_nodes[_].get_computing_capability(),
        ) for _ in range(self._edge_num)]
        
        self._cloud_computing_resource_queue = cc_ressource_queue(
            time_slot_num=self._slot_length,
            name="cc_ressource_queue",
            cloud_computing_capability=self._cloud.get_computing_capability(),
        ) 
        
        self._done = False
        
        self._resutls_file_name = "/root/HARL/results/" \
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        
        self.init_results_to_store()
        
        print("max_action: ", self._max_action)
        print("max_observation: ", self._max_observation)
        
        self._init = True
        
        self.init_maximum_cost_and_phi_t()
        
        self._init = False

        self._env_tag = "scenario_" + str(self.args["scenario"]) + "_vehicle_number_" + str(self._vehicle_num) + "_arrival_rate_" + str(self._min_task_arrival_rate_of_vehicles)

    def get_client_vehicle_num(self):
        return self._client_vehicle_num

    def get_server_vehicle_num(self):
        return self._server_vehicle_num

    def get_edge_num(self):
        return self._edge_num

    def get_maximum_task_generation_number_of_vehicles(self):
        return self._maximum_task_generation_number_of_vehicles

    def get_maximum_server_vehicle_num(self):
        return self._maximum_server_vehicle_num
    
    def get_v2v_n_components(self):
        return self._v2v_n_components
    
    def get_v2i_n_components(self):
        return self._v2i_n_components
    
    def get_maximum_task_offloaded_at_client_vehicle_number(self):
        return self._maximum_task_offloaded_at_client_vehicle_number
    
    def get_maximum_task_offloaded_at_server_vehicle_number(self):
        return self._maximum_task_offloaded_at_server_vehicle_number
    
    def get_maximum_task_offloaded_at_edge_node_number(self):
        return self._maximum_task_offloaded_at_edge_node_number
    
    def get_maximum_task_offloaded_at_cloud_number(self):
        return self._maximum_task_offloaded_at_cloud_number

    def get_v2v_connections(self):
        return self._vehicles_under_V2V_communication_range[:, :, self.cur_step]
    
    def get_v2i_connections(self):
        return self._vehicles_under_V2I_communication_range[:, :, self.cur_step]

    def collect_initial_v2v_data(self):
        '''
        收集用于拟合 V2V PCA 的初始数据
        假设您有一个方法或循环来获取多个时间步的 V2V 连接状态
        '''
        data = []

        for step in range(self._slot_length):
            # 假设您在每个时间步都能获取到 _distance_matrix_between_client_vehicles_and_server_vehicles
            v2v_connections = []
            for client in range(self._client_vehicle_num):
                for server in range(self._server_vehicle_num):
                    connection = self._distance_matrix_between_client_vehicles_and_server_vehicles[client][server][step]
                    v2v_connections.append(connection)
            data.append(v2v_connections)

        return np.array(data)

    def collect_initial_v2i_data(self):
        '''
        收集用于拟合 V2I PCA 的初始数据
        '''
        data = []

        for step in range(self._slot_length):
            v2i_connections = []
            for client in range(self._client_vehicle_num):
                for edge in range(self._edge_num):
                    connection = self._distance_matrix_between_client_vehicles_and_edge_nodes[client][edge][step]
                    v2i_connections.append(connection)
            data.append(v2i_connections)

        return np.array(data)
        
    def init_maximum_cost_and_phi_t(self):
        for _ in range(10):
            for __ in range(self._slot_length):
                self.step(self.random_action(
                    max_action=self._max_action,
                    agent_number=self.n_agents,
                ))
                
    def random_action(
        self,
        max_action : int,
        agent_number : int,
    ):
        actions = []
        for _ in range(agent_number):
            # generate the random values in the range of 0 to 1 of the size of max_action
            if self._using_PPO:
                action = np.random.uniform(-1, 1, max_action)
            else:
                action = np.random.rand(max_action)
            actions.append(action)
        return actions
    
    def init_results_to_store(self):
        # the results need to be stored
        self._reward = 0.0
        self._rewards = np.zeros((self._slot_length, ))
        self._costs = np.zeros((self._slot_length, ))
        self._lc_costs = np.zeros((self._slot_length, ))
        self._v2v_costs = np.zeros((self._slot_length, ))
        self._vc_costs = np.zeros((self._slot_length, ))
        self._v2i_costs = np.zeros((self._slot_length, ))
        self._i2i_costs = np.zeros((self._slot_length, ))
        self._ec_costs = np.zeros((self._slot_length, ))
        self._i2c_costs = np.zeros((self._slot_length, ))
        self._cc_costs = np.zeros((self._slot_length, ))
        
        self._total_delay_queues = np.zeros((self._slot_length, ))
        self._lc_delay_queues = np.zeros((self._slot_length, ))
        self._v2v_delay_queues = np.zeros((self._slot_length, ))
        self._vc_delay_queues = np.zeros((self._slot_length, ))
        self._v2i_delay_queues = np.zeros((self._slot_length, ))
        self._i2i_delay_queues = np.zeros((self._slot_length, ))
        self._ec_delay_queues = np.zeros((self._slot_length, ))
        self._i2c_delay_queues = np.zeros((self._slot_length, ))
        self._cc_delay_queues = np.zeros((self._slot_length, ))
        
        self._lc_res_queues = np.zeros((self._slot_length, ))
        self._vc_res_queues = np.zeros((self._slot_length, ))
        self._ec_res_queues = np.zeros((self._slot_length, ))
        self._cc_res_queues = np.zeros((self._slot_length, ))
        
        self._task_num_sum = 0
        self._task_nums = np.zeros((self._slot_length, ))
        self._task_uploaded_at_edge = np.zeros((self._slot_length, ))
        self._task_processed_at_local = np.zeros((self._slot_length, ))
        self._task_processed_at_vehicle = np.zeros((self._slot_length, ))
        self._task_processed_at_edge = np.zeros((self._slot_length, ))
        self._task_processed_at_local_edge = np.zeros((self._slot_length, ))
        self._task_processed_at_other_edge = np.zeros((self._slot_length, ))
        self._task_processed_at_cloud = np.zeros((self._slot_length, ))
        self._task_successfully_processed_num = np.zeros((self._slot_length, ))
        
        self._ave_task_num_of_each_client_vehicle = np.zeros((self._slot_length, ))
        self._avg_task_uploaded_at_each_edge = np.zeros((self._slot_length,))
        self._avg_task_processed_at_local = np.zeros((self._slot_length, ))
        self._avg_task_processed_at_vehicle = np.zeros((self._slot_length, ))
        self._avg_task_processed_at_edge = np.zeros((self._slot_length, ))
        self._avg_task_processed_at_local_edge = np.zeros((self._slot_length, ))
        self._avg_task_processed_at_other_edge = np.zeros((self._slot_length, ))
        self._avg_task_processed_at_cloud = np.zeros((self._slot_length, ))
    
    def obtain_task_generation_numbers(
        self,
        client_vehicles: List[vehicle],
    ):
        maximum_task_generation_number = 0
        average_task_generation_numbers = np.zeros((self._slot_length, ))
        sum_task_generation_numbers = np.zeros((self._slot_length, ))
        for t in range(self._slot_length):
            for vehicle in client_vehicles:
                sum_task_generation_numbers[t] += len(vehicle.get_tasks_by_time(t))
                average_task_generation_numbers[t] += len(vehicle.get_tasks_by_time(t))
                task_generation_number = len(vehicle.get_tasks_by_time(t))
                if task_generation_number > maximum_task_generation_number:
                    maximum_task_generation_number = task_generation_number
            average_task_generation_numbers[t] /= len(client_vehicles)
        average_number = np.mean(average_task_generation_numbers)
        maximum_task_generation_number = int((average_number * 2 + maximum_task_generation_number) / 3)
        return maximum_task_generation_number, average_task_generation_numbers, sum_task_generation_numbers
        
    def obtain_offloading_numbers(
        self,
        average_task_generation_numbers: np.ndarray,
        sum_task_generation_numbers: np.ndarray,
    ):
        maximum_task_offloaded_at_client_vehicle_number = 0
        maximum_task_offloaded_at_server_vehicle_number = 0
        maximum_task_offloaded_at_edge_node_number = 0
        maximum_task_offloaded_at_cloud_number = 0
        tasks_num = 0
        maximum_sum = 0
        average_vehicle_task_num = 0
        for t in range(self._slot_length):
            tasks_num += sum_task_generation_numbers[t]
            average_vehicle_task_num += average_task_generation_numbers[t]
            if sum_task_generation_numbers[t] > maximum_sum:
                maximum_sum = sum_task_generation_numbers[t]
        average_sum = tasks_num / self._slot_length
        average_vehicle_task_num /= self._slot_length
        
        # print("task_num: ", tasks_num)
        # print("average_sum: ", average_sum)
        # print("maximum_sum: ", maximum_sum)
        # print("average_vehicle_task_num: ", average_vehicle_task_num)
        
        maximum_task_offloaded_at_client_vehicle_number = int(average_vehicle_task_num * 0.5)
        if maximum_task_offloaded_at_client_vehicle_number == 0:
            maximum_task_offloaded_at_client_vehicle_number = 1
        
        maximum_task_offloaded_at_server_vehicle_number = int(average_vehicle_task_num * 2)
        if maximum_task_offloaded_at_server_vehicle_number == 0:
            maximum_task_offloaded_at_server_vehicle_number = 1
        
        maximum_task_offloaded_at_edge_node_number = int((average_sum * 0.10 + maximum_sum  * 0.10) / 2)
        if maximum_task_offloaded_at_edge_node_number == 0:
            maximum_task_offloaded_at_edge_node_number = 1
            
        maximum_task_offloaded_at_cloud_number = int((average_sum * 0.20 + maximum_sum  * 0.20) / 2)
        if maximum_task_offloaded_at_cloud_number == 0:
            maximum_task_offloaded_at_cloud_number = 1
        
        # print("maximum_task_offloaded_at_client_vehicle_number: ", maximum_task_offloaded_at_client_vehicle_number)
        # print("maximum_task_offloaded_at_server_vehicle_number: ", maximum_task_offloaded_at_server_vehicle_number)
        # print("maximum_task_offloaded_at_edge_node_number: ", maximum_task_offloaded_at_edge_node_number)
        # print("maximum_task_offloaded_at_cloud_number: ", maximum_task_offloaded_at_cloud_number)
        
        return maximum_task_offloaded_at_client_vehicle_number, maximum_task_offloaded_at_server_vehicle_number, maximum_task_offloaded_at_edge_node_number, maximum_task_offloaded_at_cloud_number
        
    def is_done(self):
        return self._done
    
    def get_n_agents(self):
        return self.n_agents
    
    def get_max_action(self):
        return self._max_action
        
    def obtain_maximum_task_data_size_and_required_cycles_of_vehicles(self, vehicles: List[vehicle]):
        maximum_task_data_size = 0.0
        maximum_task_required_cycles = 0.0
        for vehicle in vehicles:
            tasks = vehicle.get_tasks()
            for task in tasks:
                if task[2].get_input_data_size() > maximum_task_data_size:
                    maximum_task_data_size = task[2].get_input_data_size()
                if task[2].get_requested_computing_cycles() > maximum_task_required_cycles:
                    maximum_task_required_cycles = task[2].get_requested_computing_cycles()
        return maximum_task_data_size, maximum_task_required_cycles

    def step(self, actions):
        # transform the actions into the task offloading actions, transmission power allocation actions, computation resource allocation actions
        
        # if not self._init:
        #     print("*" * 50)
        #     print("cur_step: ", self.cur_step)
        
        # print("vehicles_under_V2V_communication_range: ", self._vehicles_under_V2V_communication_range)
        # print("vehicles_under_V2I_communication_range: ", self._vehicles_under_V2V_communication_range)
        # now_time = time.time()
        
        # print("actions: ", actions) 

        if self._using_PPO:
            processed_actions = self.process_actions(actions)   # covert the actions into the value of [0, 1]
        else:
            processed_actions = actions

        # print("processed_actions: ", processed_actions)
        
        task_offloading_actions, transmission_power_allocation_actions, \
            computation_resource_allocation_actions = self.transform_actions(
                actions=processed_actions,
                vehicles_under_V2V_communication_range=self._vehicles_under_V2V_communication_range,)
        
        # print("task_offloading_actions: ", task_offloading_actions)
        # print("transmission_power_allocation_actions: ", transmission_power_allocation_actions)
        # print("computation_resource_allocation_actions: ", computation_resource_allocation_actions)

        # end_time = time.time()
        # print("transform_actions time: ", end_time - now_time)
        
        # now_time = time.time()
        self._task_offloaded_at_client_vehicles, self._task_offloaded_at_server_vehicles, \
            self._task_offloaded_at_edge_nodes, self._task_uploaded_at_edge_nodes, \
                self._task_offloaded_at_cloud = self.obtain_tasks_offloading_conditions(
                task_offloading_actions=task_offloading_actions,
                vehicles_under_V2V_communication_range=self._vehicles_under_V2V_communication_range,
                vehicles_under_V2I_communication_range=self._vehicles_under_V2I_communication_range,
                now = self.cur_step,
            )
        # print("task_offloaded_at_client_vehicles: ", self._task_offloaded_at_client_vehicles)
        # print("task_offloaded_at_server_vehicles: ", self._task_offloaded_at_server_vehicles)
        # print("task_offloaded_at_edge_nodes: ", self._task_offloaded_at_edge_nodes)
        # print("task_offloaded_at_cloud: ", self._task_offloaded_at_cloud)
        # end_time = time.time()
        # print("obtain_tasks_offloading_conditions time: ", end_time - now_time)    
        self.update_self_queues()
        
        # now_time = time.time()
        self.update_actural_queues(
            transmission_power_allocation_actions=transmission_power_allocation_actions,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        # end_time = time.time()
        # print("update_actural_queues time: ", end_time - now_time)
        
        # now_time = time.time()
        self.update_virtual_queues(
            task_offloading_actions=task_offloading_actions,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        # now_time = time.time()
        reward = self.compute_reward(
            task_offloading_actions=task_offloading_actions,
            transmission_power_allocation_actions=transmission_power_allocation_actions,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        
        self._reward += reward
        self._rewards[self.cur_step] = reward
        # if not self._init:
        #     print("reward: ", reward)
        
        # end_time = time.time()
        # print("update_virtual_queues time: ", end_time - now_time)

        # end_time = time.time()
        # print("compute_reward time: ", end_time - now_time)
        
        # update the environment
        # now_time = time.time()
        self.cur_step += 1
        
        if self.cur_step >= self._slot_length:
            self.cur_step = self._slot_length
        
        dones = [False for _ in range(self.n_agents)]
        info = [{"bad_transition": False} for _ in range(self.n_agents)]
        if self.cur_step == self._slot_length:
            for _ in range(self.n_agents):
                info[_]["bad_transition"] = True
            dones = [True for _ in range(self.n_agents)]
            self._done = True
        
        if self._done:
            if not self._init:
                self.save_results()
                self._episode_index += 1
            self.reset()
            
        obs = self.obtain_observation()
        env_state = self.state()
        s_obs = self.repeat(env_state)
        rewards = [[reward]] * self.n_agents
        # end_time = time.time()
        # print("step time: ", end_time - now_time)
        
        return (
            obs,
            s_obs,
            rewards,
            dones,
            info,
            self.get_avail_actions(),
        )

    def get_reward(self):
        return self._rewards[self.cur_step]
        
    def update_self_queues(self):
        delay_queue = 0.0
        lc_delay_queue = 0.0
        v2v_delay_queue = 0.0
        vc_delay_queue = 0.0
        v2i_delay_queue = 0.0
        i2i_delay_queue = 0.0
        ec_delay_queue = 0.0
        i2c_delay_queue = 0.0
        cc_delay_queue = 0.0
        
        lc_res_queue = 0.0
        vc_res_queue = 0.0
        ec_res_queue = 0.0
        cc_res_queue = 0.0
        
        for _ in range(self._client_vehicle_num):
            delay_queue += self._delay_queues[_].get_queue(time_slot=self.cur_step)
            lc_delay_queue += self._lc_queues[_].get_queue(time_slot=self.cur_step)
            lc_res_queue += self._local_computing_resource_queues[_].get_queue(time_slot=self.cur_step)
            
        delay_queue /= self._client_vehicle_num
        lc_delay_queue /= self._client_vehicle_num
        lc_res_queue /= self._client_vehicle_num
        
        for _ in range(self._server_vehicle_num):
            v2v_delay_queue += self._v2v_queues[_].get_queue(time_slot=self.cur_step)
            vc_delay_queue += self._vc_queues[_].get_queue(time_slot=self.cur_step)
            vc_res_queue += self._vehicle_computing_resource_queues[_].get_queue(time_slot=self.cur_step)
        
        v2v_delay_queue /= self._server_vehicle_num
        vc_delay_queue /= self._server_vehicle_num
        vc_res_queue /= self._server_vehicle_num
        
        for _ in range(self._edge_num):
            v2i_delay_queue += self._v2i_queues[_].get_queue(time_slot=self.cur_step)
            i2i_delay_queue += self._i2i_queues[_].get_queue(time_slot=self.cur_step)
            ec_delay_queue += self._ec_queues[_].get_queue(time_slot=self.cur_step)
            ec_res_queue += self._edge_computing_resource_queues[_].get_queue(time_slot=self.cur_step)
        
        v2i_delay_queue /= self._edge_num
        i2i_delay_queue /= self._edge_num
        ec_delay_queue /= self._edge_num
        ec_res_queue /= self._edge_num
        
        i2c_delay_queue = self._i2c_quque.get_queue(time_slot=self.cur_step)
        cc_delay_queue = self._cc_queue.get_queue(time_slot=self.cur_step)
        cc_res_queue = self._cloud_computing_resource_queue.get_queue(time_slot=self.cur_step)
        
        self._total_delay_queues[self.cur_step] = delay_queue
        self._lc_delay_queues[self.cur_step] = lc_delay_queue
        self._v2v_delay_queues[self.cur_step] = v2v_delay_queue
        self._vc_delay_queues[self.cur_step] = vc_delay_queue
        self._v2i_delay_queues[self.cur_step] = v2i_delay_queue
        self._i2i_delay_queues[self.cur_step] = i2i_delay_queue
        self._ec_delay_queues[self.cur_step] = ec_delay_queue
        self._i2c_delay_queues[self.cur_step] = i2c_delay_queue
        self._cc_delay_queues[self.cur_step] = cc_delay_queue
        
        self._lc_res_queues[self.cur_step] = lc_res_queue
        self._vc_res_queues[self.cur_step] = vc_res_queue
        self._ec_res_queues[self.cur_step] = ec_res_queue
        self._cc_res_queues[self.cur_step] = cc_res_queue
        
        
    def reset(self):
        self._done = False
        self._seed += 1
        self.cur_step = 0
        
        # reset the environment
                
        self._task_offloaded_at_client_vehicles = {}
        self._task_offloaded_at_server_vehicles = {}
        self._task_offloaded_at_edge_nodes = {}
        self._task_uploaded_at_edge_nodes = {}
        self._task_offloaded_at_cloud = {}
        
        for i in range(self._client_vehicle_num):
            self._task_offloaded_at_client_vehicles["client_vehicle_" + str(i)] = []
        for i in range(self._server_vehicle_num):
            self._task_offloaded_at_server_vehicles["server_vehicle_" + str(i)] = []
        for i in range(self._edge_num):
            self._task_offloaded_at_edge_nodes["edge_node_" + str(i)] = []
        self._task_offloaded_at_cloud["cloud"] = []
        
        for _ in range(self._client_vehicle_num):
            self._lc_queues[_].reset()
        for _ in range(self._server_vehicle_num):
            self._v2v_queues[_].reset()
            self._vc_queues[_].reset()
        for _ in range(self._edge_num):
            self._v2i_queues[_].reset()
            self._i2i_queues[_].reset()
            self._ec_queues[_].reset()
        self._i2c_quque.reset()
        self._cc_queue.reset()
        
        self._lc_queue_backlogs = np.zeros((self._client_vehicle_num, self._slot_length))
        self._v2v_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._vc_queue_backlogs = np.zeros((self._server_vehicle_num, self._slot_length))
        self._v2i_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._i2i_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._ec_queue_backlogs = np.zeros((self._edge_num, self._slot_length))
        self._i2c_queue_backlogs = np.zeros((self._slot_length, ))
        self._cc_queue_backlogs = np.zeros((self._slot_length, ))
        
        for _ in range(self._client_vehicle_num):
            self._delay_queues[_].reset()
        for _ in range(self._client_vehicle_num):
            self._local_computing_resource_queues[_].reset()
        for _ in range(self._server_vehicle_num):
            self._vehicle_computing_resource_queues[_].reset()
        for _ in range(self._edge_num):
            self._edge_computing_resource_queues[_].reset()
        self._cloud_computing_resource_queue.reset()
        
        self.init_results_to_store()
        
        obs = self.obtain_observation()
        s_obs = self.repeat(self.state())
        
        return obs, s_obs, self.get_avail_actions()
        
    
    def update_actural_queues(
        self,
        transmission_power_allocation_actions: Dict,
        computation_resource_allocation_actions: Dict,
    ):
        if self.cur_step < self._slot_length - 1:
            # update the lc queues
            
            # now_time = time.time()
            for client_vehicle_index in range(self._client_vehicle_num):    
                lc_queue_input = self._lc_queues[client_vehicle_index].compute_input(
                    client_vehicle=self._client_vehicles[client_vehicle_index],
                    task_offloaded_at_client_vehicles=self._task_offloaded_at_client_vehicles,
                )
                lc_queue_output = self._lc_queues[client_vehicle_index].compute_output(
                    computation_resource_allocation_actions=computation_resource_allocation_actions,
                    task_offloaded_at_client_vehicles=self._task_offloaded_at_client_vehicles,
                    client_vehicle_computing_capability=self._client_vehicles[client_vehicle_index].get_computing_capability(),
                )
                # print("lc_queue_input: ", lc_queue_input)
                # print("lc_queue_output: ", lc_queue_output)
                
                self._lc_queues[client_vehicle_index].update(
                    input=lc_queue_input,
                    output=lc_queue_output,
                    time_slot=self.cur_step,
                )
                self._lc_queue_backlogs[client_vehicle_index][self.cur_step + 1] = self._lc_queues[client_vehicle_index].get_queue(time_slot=self.cur_step + 1)
                if self._lc_queues[client_vehicle_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_lc_queue_length[client_vehicle_index]:
                    self._maximum_lc_queue_length[client_vehicle_index] = self._lc_queues[client_vehicle_index].get_queue(time_slot=self.cur_step + 1)
            # end_time = time.time()
            # print("update_lc_queues time: ", end_time - now_time)
            
            # update the v2v and vc queues
            # v2v_vc_now_time = time.time()
            # v2v_time = 0.0
            # v2v_input_time = 0.0
            # v2v_output_time = 0.0
            # v2v_update_time = 0.0
            # vc_time = 0.0
            for server_vehicle_index in range(self._server_vehicle_num):
                # v2v_now_time = time.time()
                # now_time = time.time()
                v2v_queue_input = self._v2v_queues[server_vehicle_index].compute_input(
                    task_offloaded_at_server_vehicles=self._task_offloaded_at_server_vehicles,
                )
                # end_time = time.time()
                # v2v_input_time += end_time - now_time
                
                # now_time = time.time()
                v2v_queue_output = self._v2v_queues[server_vehicle_index].compute_output(
                    task_offloaded_at_server_vehicles=self._task_offloaded_at_server_vehicles,
                    transmission_power_allocation_actions=transmission_power_allocation_actions,
                    now=self.cur_step,
                )
                # end_time = time.time()
                # v2v_output_time += end_time - now_time
                
                # print("v2v_queue_input: ", v2v_queue_input)
                # print("v2v_queue_output: ", v2v_queue_output)
                # now_time = time.time()
                self._v2v_queues[server_vehicle_index].update(
                    input=v2v_queue_input,
                    output=v2v_queue_output,
                    time_slot=self.cur_step,
                )
                # end_time = time.time()
                # v2v_update_time += end_time - now_time
                
                # print("*" * 100)
                # print("self._v2v_queue_backlogs:", self._v2v_queue_backlogs)
                # print("after")
                self._v2v_queue_backlogs[server_vehicle_index][self.cur_step + 1] = self._v2v_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1)
                
                
                # print("self._v2v_queue_backlogs:", self._v2v_queue_backlogs)
                if self._v2v_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_v2v_queue_length[server_vehicle_index]:
                    self._maximum_v2v_queue_length[server_vehicle_index] = self._v2v_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1)
                # v2v_end_time = time.time()
                # v2v_time += v2v_end_time - v2v_now_time
                
                # now_time = time.time()
                vc_queue_input = self._vc_queues[server_vehicle_index].compute_input(
                    v2v_transmission_input=v2v_queue_input,
                    v2v_transmission_output=v2v_queue_output,
                )
                vc_queue_output = self._vc_queues[server_vehicle_index].compute_output(
                    server_vehicle_compute_capability=self._server_vehicles[server_vehicle_index].get_computing_capability(),
                    computation_resource_allocation_actions=computation_resource_allocation_actions,
                    task_offloaded_at_server_vehicles=self._task_offloaded_at_server_vehicles,
                )
                # print("vc_queue_input: ", vc_queue_input)
                # print("vc_queue_output: ", vc_queue_output)
                self._vc_queues[server_vehicle_index].update(
                    input=vc_queue_input,
                    output=vc_queue_output,
                    time_slot=self.cur_step,
                )
                self._vc_queue_backlogs[server_vehicle_index][self.cur_step + 1] = self._vc_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1)
                if self._vc_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_vc_queue_length[server_vehicle_index]:
                    self._maximum_vc_queue_length[server_vehicle_index] = self._vc_queues[server_vehicle_index].get_queue(time_slot=self.cur_step + 1)
            #     end_time = time.time()
            #     vc_time += end_time - now_time
            # v2v_vc_end_time = time.time()
            # print("v2v_input_time: ", v2v_input_time)
            # print("v2v_output_time: ", v2v_output_time)
            # print("v2v_update_time: ", v2v_update_time)
            # print("update_v2v_queues time: ", v2v_time)
            # print("update_vc_queues time: ", vc_time)
            # print("update_v2v_vc_queues time: ", v2v_vc_end_time - v2v_vc_now_time)
            
            # update the v2i, i2i and ec queues
            # v2i_i2i_ec_now_time = time.time()
            # v2i_time = 0.0
            # i2i_time = 0.0
            # ec_time = 0.0
            for edge_node_index in range(self._edge_num):
                # now_time = time.time()
                v2i_queue_input = self._v2i_queues[edge_node_index].compute_input(
                    task_uploaded_at_edge_nodes=self._task_uploaded_at_edge_nodes,
                )
                v2i_queue_output = self._v2i_queues[edge_node_index].compute_output(
                    task_uploaded_at_edge_nodes=self._task_uploaded_at_edge_nodes,
                    transmission_power_allocation_actions=transmission_power_allocation_actions,
                    now=self.cur_step,
                )
                # print("v2i_queue_input: ", v2i_queue_input)
                # print("v2i_queue_output: ", v2i_queue_output)
                self._v2i_queues[edge_node_index].update(
                    input=v2i_queue_input,
                    output=v2i_queue_output,
                    time_slot=self.cur_step,
                )
                self._v2i_queue_backlogs[edge_node_index][self.cur_step + 1] = self._v2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
                if self._v2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_v2i_queue_length[edge_node_index]:
                    self._maximum_v2i_queue_length[edge_node_index] = self._v2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
                # end_time = time.time()
                # v2i_time += end_time - now_time
                
                # now_time = time.time()                  
                i2i_queue_input = self._i2i_queues[edge_node_index].compute_input(
                    task_offloaded_at_edge_nodes=self._task_offloaded_at_edge_nodes,
                    task_uploaded_at_edge_nodes=self._task_uploaded_at_edge_nodes,
                    transmission_power_allocation_actions=transmission_power_allocation_actions,
                    now=self.cur_step,
                )
                i2i_queue_output = self._i2i_queues[edge_node_index].compute_output(
                    task_offloaded_at_edge_nodes=self._task_offloaded_at_edge_nodes,
                    distance_matrix_between_edge_nodes=self._distance_matrix_between_edge_nodes,
                )
                # print("i2i_queue_input: ", i2i_queue_input)
                # print("i2i_queue_output: ", i2i_queue_output)
                self._i2i_queues[edge_node_index].update(
                    input=i2i_queue_input,
                    output=i2i_queue_output,
                    time_slot=self.cur_step,
                )
                self._i2i_queue_backlogs[edge_node_index][self.cur_step + 1] = self._i2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
                if self._i2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_i2i_queue_length[edge_node_index]:
                    self._maximum_i2i_queue_length[edge_node_index] = self._i2i_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
                # end_time = time.time()
                # i2i_time += end_time - now_time
                
                # now_time = time.time()
                ec_queue_input = self._ec_queues[edge_node_index].compute_input(
                    v2i_transmission_input=v2i_queue_input,
                    v2i_transmission_output=v2i_queue_output,
                    i2i_transmission_input=i2i_queue_input,
                    i2i_transmission_output=i2i_queue_output,
                )
                ec_queue_output = self._ec_queues[edge_node_index].compute_output(
                    edge_node_computing_capability=self._edge_nodes[edge_node_index].get_computing_capability(),
                    computation_resource_allocation_actions=computation_resource_allocation_actions,
                    task_offloaded_at_edge_nodes=self._task_offloaded_at_edge_nodes,
                )
                # print("ec_queue_input: ", ec_queue_input)
                # print("ec_queue_output: ", ec_queue_output)
                self._ec_queues[edge_node_index].update(
                    input=ec_queue_input,
                    output=ec_queue_output,
                    time_slot=self.cur_step,
                )
                self._ec_queue_backlogs[edge_node_index][self.cur_step + 1] = self._ec_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
                if self._ec_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1) > self._maximum_ec_queue_length[edge_node_index]:
                    self._maximum_ec_queue_length[edge_node_index] = self._ec_queues[edge_node_index].get_queue(time_slot=self.cur_step + 1)
            #     end_time = time.time()
            #     ec_time += end_time - now_time
                
            # v2i_i2i_ec_end_time = time.time()
            # print("v2i_time: ", v2i_time)
            # print("i2i_time: ", i2i_time)
            # print("ec_time: ", ec_time)
            # print("update_v2i_i2i_ec_queues time: ", v2i_i2i_ec_end_time - v2i_i2i_ec_now_time)
            
            # update the i2c and cc queues
            # now_time = time.time()
            i2c_queue_input = self._i2c_quque.compute_input(
                task_offloaded_at_cloud=self._task_offloaded_at_cloud,
                task_uploaded_at_edge_nodes=self._task_uploaded_at_edge_nodes,
                transmission_power_allocation_actions=transmission_power_allocation_actions,
                now=self.cur_step,
            )
            # end_time = time.time()
            # print("i2c compute_input time: ", end_time - now_time)
            # now_time = time.time()
            i2c_queue_output = self._i2c_quque.compute_output(
                task_offloaded_at_cloud=self._task_offloaded_at_cloud,
                distance_matrix_between_edge_nodes_and_the_cloud=self._distance_matrix_between_edge_nodes_and_the_cloud,
            )
            # print("i2c_queue_input: ", i2c_queue_input)
            # print("i2c_queue_output: ", i2c_queue_output)
            # end_time = time.time()
            # print("i2c compute_output time: ", end_time - now_time)
            # now_time = time.time()
            self._i2c_quque.update(
                input=i2c_queue_input,
                output=i2c_queue_output,
                time_slot=self.cur_step,
            )
            self._i2c_queue_backlogs[self.cur_step + 1] = self._i2c_quque.get_queue(time_slot=self.cur_step + 1)
            if self._i2c_quque.get_queue(time_slot=self.cur_step + 1) > self._maximum_i2c_queue_length:
                self._maximum_i2c_queue_length = self._i2c_quque.get_queue(time_slot=self.cur_step + 1)
            # end_time = time.time()
            # print("update_i2c_queue time: ", end_time - now_time)
            
            # now_time = time.time()
            cc_queue_input = self._cc_queue.compute_input(
                i2c_transmission_input=i2c_queue_input,
                i2c_transmission_output=i2c_queue_output,
            )
            cc_queue_output = self._cc_queue.compute_output(
                cloud_computing_capability=self._cloud.get_computing_capability(),
                computation_resource_allocation_actions=computation_resource_allocation_actions,
                task_offloaded_at_cloud=self._task_offloaded_at_cloud,
            )
            # print("cc_queue_input: ", cc_queue_input)
            # print("cc_queue_output: ", cc_queue_output)
            self._cc_queue.update(
                input=cc_queue_input,
                output=cc_queue_output,
                time_slot=self.cur_step,
            )
            self._cc_queue_backlogs[self.cur_step + 1] = self._cc_queue.get_queue(time_slot=self.cur_step + 1)
            if self._cc_queue.get_queue(time_slot=self.cur_step + 1) > self._maximum_cc_queue_length:
                self._maximum_cc_queue_length = self._cc_queue.get_queue(time_slot=self.cur_step + 1)
            # end_time = time.time()
            # print("update_cc_queue time: ", end_time - now_time)
                    
    def update_virtual_queues(
        self,
        task_offloading_actions: Dict,
        computation_resource_allocation_actions: Dict,
    ):
        # print("lc_queue_backlogs: ", self._lc_queue_backlogs)
        # print("v2v_queue_backlogs: ", self._v2v_queue_backlogs)
        # print("vc_queue_backlogs: ", self._vc_queue_backlogs)
        # print("v2i_queue_backlogs: ", self._v2i_queue_backlogs)
        # print("i2i_queue_backlogs: ", self._i2i_queue_backlogs)
        # print("ec_queue_backlogs: ", self._ec_queue_backlogs)
        # print("i2c_queue_backlogs: ", self._i2c_queue_backlogs)
        # print("cc_queue_backlogs: ", self._cc_queue_backlogs)
        
        # update the delay queues
        for client_vehicle_index in range(self._client_vehicle_num):
            delay_queue_input = self._delay_queues[client_vehicle_index].compute_input(
                client_vehicles=self._client_vehicles,
                now=self.cur_step,
                task_offloading_decisions=task_offloading_actions,
                lc_queue_backlogs=self._lc_queue_backlogs,
                v2v_queue_backlogs=self._v2v_queue_backlogs,
                vc_queue_backlogs=self._vc_queue_backlogs,
                v2i_queue_backlogs=self._v2i_queue_backlogs,
                i2i_queue_backlogs=self._i2i_queue_backlogs,
                ec_queue_backlogs=self._ec_queue_backlogs,
                i2c_queue_backlogs=self._i2c_queue_backlogs,
                cc_queue_backlogs=self._cc_queue_backlogs,
            )
            delay_queue_output = self._delay_queues[client_vehicle_index].compute_output(
                now=self.cur_step,
                client_vehicles=self._client_vehicles,
            )
            # print("delay_queue_input: ", delay_queue_input)
            # print("delay_queue_output: ", delay_queue_output)
            self._delay_queues[client_vehicle_index].update(
                input=delay_queue_input,
                output=delay_queue_output,
                time_slot=self.cur_step,
            )
        
        # update the local computing resource queues
        for client_vehicle_index in range(self._client_vehicle_num):
            lc_ressource_queue_input = self._local_computing_resource_queues[client_vehicle_index].compute_input(
                task_offloaded_at_client_vehicles=self._task_offloaded_at_client_vehicles,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            lc_ressource_queue_output = self._local_computing_resource_queues[client_vehicle_index].compute_output()
            self._local_computing_resource_queues[client_vehicle_index].update(
                input=lc_ressource_queue_input,
                output=lc_ressource_queue_output,
                time_slot=self.cur_step,
            )
            # print("lc_ressource_queue_input: ", lc_ressource_queue_input)
            # print("lc_ressource_queue_output: ", lc_ressource_queue_output)
        
        # update the vehicle computing resource queues
        for server_vehicle_index in range(self._server_vehicle_num):
            vc_ressource_queue_input = self._vehicle_computing_resource_queues[server_vehicle_index].compute_input(
                task_offloaded_at_server_vehicles=self._task_offloaded_at_server_vehicles,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            vc_ressource_queue_output = self._vehicle_computing_resource_queues[server_vehicle_index].compute_output()
            self._vehicle_computing_resource_queues[server_vehicle_index].update(
                input=vc_ressource_queue_input,
                output=vc_ressource_queue_output,
                time_slot=self.cur_step,
            )
            # print("vc_ressource_queue_input: ", vc_ressource_queue_input)
            # print("vc_ressource_queue_output: ", vc_ressource_queue_output)
        
        # update the edge computing resource queues
        for edge_node_index in range(self._edge_num):
            ec_ressource_queue_input = self._edge_computing_resource_queues[edge_node_index].compute_input(
                task_offloaded_at_edge_nodes=self._task_offloaded_at_edge_nodes,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            ec_ressource_queue_output = self._edge_computing_resource_queues[edge_node_index].compute_output()
            self._edge_computing_resource_queues[edge_node_index].update(
                input=ec_ressource_queue_input,
                output=ec_ressource_queue_output,
                time_slot=self.cur_step,
            )
            # print("ec_ressource_queue_input: ", ec_ressource_queue_input)
            # print("ec_ressource_queue_output: ", ec_ressource_queue_output)
        
        # update the cloud computing resource queue
        cc_ressource_queue_input = self._cloud_computing_resource_queue.compute_input(
            task_offloaded_at_cloud=self._task_offloaded_at_cloud,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        cc_ressource_queue_output = self._cloud_computing_resource_queue.compute_output()
        self._cloud_computing_resource_queue.update(
            input=cc_ressource_queue_input,
            output=cc_ressource_queue_output,
            time_slot=self.cur_step,
        )
        # print("cc_ressource_queue_input: ", cc_ressource_queue_input)
        # print("cc_ressource_queue_output: ", cc_ressource_queue_output)
    
    def compute_reward(
        self,
        task_offloading_actions: Dict,
        transmission_power_allocation_actions: Dict,
        computation_resource_allocation_actions: Dict,
    ):        
        # print("task_offloading_actions: ", task_offloading_actions)
        # print("transmission_power_allocation_actions: ", transmission_power_allocation_actions)
        # print("computation_resource_allocation_actions: ", computation_resource_allocation_actions)
        
        total_cost = self.compute_total_cost(
            task_offloading_actions=task_offloading_actions,
            transmission_power_allocation_actions=transmission_power_allocation_actions,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        
        self._costs[self.cur_step] = total_cost
        
        # print("total_cost: ", total_cost)
        
        # print("delay_queues: ", self._delay_queues)
        # print("lc_queues: ", self._local_computing_resource_queues)
        # print("vc_queues: ", self._vehicle_computing_resource_queues)
        # print("ec_queues: ", self._edge_computing_resource_queues)
        
        phi_t = self.compute_total_phi_t()
        
        # print("phi_t: ", phi_t)
        
        reward = self.transform_reward(
            total_cost=total_cost,
            phi_t=phi_t,
            penalty_weight=self._penalty_weight,
        )
        
        return reward
    
    def transform_reward(
        self,
        total_cost: float,
        phi_t: float,
        penalty_weight: float,
    ):
        # 确保 total_cost 和 phi_t 大于 0
        total_cost = max(total_cost, 1e-6)
        phi_t = max(phi_t, 1e-6)

        # 计算成功处理任务的比例
        task_success_rate = self._task_successfully_processed_num[self.cur_step] / max(self._task_nums[self.cur_step], 1e-6)

        # 增强 task_success_rate 的影响
        gamma = 1  # 可以根据需要调整
        enhanced_task_success_rate = task_success_rate ** gamma

        # 归一化
        cost_term = -np.log(total_cost * penalty_weight)
        phi_term = -np.log(phi_t)

        # 定义权重系数
        alpha = 0.7  # 控制 cost_term 和 phi_term 的权重
        beta = 0.5   # 控制 enhanced_task_success_rate 的权重

        # 计算未归一化的奖励
        reward = beta * enhanced_task_success_rate + (1 - beta) * (alpha * cost_term + (1 - alpha) * phi_term)

        # 初始化最小值和最大值
        if self.min_reward is None:
            self.min_reward = reward
            self.max_reward = reward
        else:
            # 使用移动平均更新最小值和最大值
            self.min_reward = self.reward_alpha * reward + (1 - self.reward_alpha) * self.min_reward
            self.max_reward = self.reward_alpha * reward + (1 - self.reward_alpha) * self.max_reward

        # 确保最小值不大于最大值
        if self.max_reward == self.min_reward:
            reward_normalized = 0.5  # 当无法归一化时，设置为中间值
        else:
            # 对奖励进行归一化处理
            reward_normalized = (reward - self.min_reward) / (self.max_reward - self.min_reward + 1e-8)
            reward_normalized = np.clip(reward_normalized, 0, 1)

        return reward
    
    def compute_total_phi_t(
        self,
    ):
        phi_t = 0.0
        now = self.cur_step
        
        phi_t_1 = 0.0
        phi_t_2 = 0.0
        phi_t_3 = 0.0
        phi_t_4 = 0.0
        phi_t_5 = 0.0
    
        for i in range(self._client_vehicle_num):
            # print("delay_queues[i].get_queue(now): ", self._delay_queues[i].get_queue(now))
            # print("delay_queues[i].get_output_by_time(now): ", self._delay_queues[i].get_output_by_time(now))
            # print("delay_queues[i].get_input_by_time(now): ", self._delay_queues[i].get_input_by_time(now))
            # print("result: ", (delay_queues[i].get_queue(now) - delay_queues[i].get_output_by_time(now) ) * delay_queues[i].get_input_by_time(now))
            phi_t_delay_queue = (self._delay_queues[i].get_queue(now) - self._delay_queues[i].get_output_by_time(now)) * \
                self._delay_queues[i].get_input_by_time(now)
            # print("phi_t_delay_queue: ", phi_t_delay_queue)
            # print("self._maximum_phi_t_delay_queue[i]: ", self._maximum_phi_t_delay_queue[i])
            if self._maximum_phi_t_delay_queue[i] < phi_t_delay_queue:
                self._maximum_phi_t_delay_queue[i] = phi_t_delay_queue
            if self._minimum_phi_t_delay_queue[i] > phi_t_delay_queue:
                self._minimum_phi_t_delay_queue[i] = phi_t_delay_queue
            if phi_t_delay_queue < 0:
                if self._minimum_phi_t_delay_queue[i] == np.inf:
                    phi_t_1 += 1
                else:
                    phi_t_1 += phi_t_delay_queue / self._minimum_phi_t_delay_queue[i]
            elif phi_t_delay_queue > 0:
                if self._maximum_phi_t_delay_queue[i] == 0:
                    phi_t_1 += 1
                else:
                    phi_t_1 += phi_t_delay_queue / self._maximum_phi_t_delay_queue[i]
            else:
                phi_t_1 += 0

        for i in range(self._client_vehicle_num):
            # print("lc_queues[i].get_queue(now): ", lc_queues[i].get_queue(now))
            # print("lc_queues[i].get_output_by_time(now): ", lc_queues[i].get_output_by_time(now))
            # print("lc_queues[i].get_input_by_time(now): ", lc_queues[i].get_input_by_time(now))
            phi_t_lc_queue = (self._lc_queues[i].get_queue(now) - self._lc_queues[i].get_output_by_time(now)) * \
                self._lc_queues[i].get_input_by_time(now)
                
            if self._maximum_phi_t_lc_queue[i] < phi_t_lc_queue:
                self._maximum_phi_t_lc_queue[i] = phi_t_lc_queue
            if self._minimum_phi_t_lc_queue[i] > phi_t_lc_queue:
                self._minimum_phi_t_lc_queue[i] = phi_t_lc_queue
            if phi_t_lc_queue < 0:
                if self._minimum_phi_t_lc_queue[i] == np.inf:
                    phi_t_2 += 1
                else:
                    phi_t_2 += phi_t_lc_queue / self._minimum_phi_t_lc_queue[i]
            elif phi_t_lc_queue > 0:
                if self._maximum_phi_t_lc_queue[i] == 0:
                    phi_t_2 += 1
                else:
                    phi_t_2 += phi_t_lc_queue / self._maximum_phi_t_lc_queue[i]
            else:
                phi_t_2 += 0

        for i in range(self._server_vehicle_num):
            # print("vc_queues[i].get_queue(now): ", vc_queues[i].get_queue(now))
            # print("vc_queues[i].get_output_by_time(now): ", vc_queues[i].get_output_by_time(now))
            # print("vc_queues[i].get_input_by_time(now): ", vc_queues[i].get_input_by_time(now))
            phi_t_vc_queue = (self._vc_queues[i].get_queue(now) - self._vc_queues[i].get_output_by_time(now)) * \
                self._vc_queues[i].get_input_by_time(now)
            if self._maximum_phi_t_vc_queue[i] < phi_t_vc_queue:
                self._maximum_phi_t_vc_queue[i] = phi_t_vc_queue
            if self._minimum_phi_t_vc_queue[i] > phi_t_vc_queue:
                self._minimum_phi_t_vc_queue[i] = phi_t_vc_queue
            
            if phi_t_vc_queue < 0:
                if self._minimum_phi_t_vc_queue[i] == np.inf:
                    phi_t_3 += 1
                else:
                    phi_t_3 += phi_t_vc_queue / self._minimum_phi_t_vc_queue[i]
            elif phi_t_vc_queue > 0:
                if self._maximum_phi_t_vc_queue[i] == 0:
                    phi_t_3 += 1
                else:
                    phi_t_3 += phi_t_vc_queue / self._maximum_phi_t_vc_queue[i]
            else:
                phi_t_3 += 0

        for i in range(self._edge_num):
            # print("ec_queues[i].get_queue(now): ", ec_queues[i].get_queue(now))
            # print("ec_queues[i].get_output_by_time(now): ", ec_queues[i].get_output_by_time(now))
            # print("ec_queues[i].get_input_by_time(now): ", ec_queues[i].get_input_by_time(now))
            phi_t_ec_queue = (self._ec_queues[i].get_queue(now) - self._ec_queues[i].get_output_by_time(now)) * \
                self._ec_queues[i].get_input_by_time(now)
            # print("phi_t_ec_queue: ", phi_t_ec_queue)
            # print("self._maximum_phi_t_ec_queue[i]: ", self._maximum_phi_t_ec_queue[i])
            if self._maximum_phi_t_ec_queue[i] < phi_t_ec_queue:
                self._maximum_phi_t_ec_queue[i] = phi_t_ec_queue
            if self._minimum_phi_t_ec_queue[i] > phi_t_ec_queue:
                self._minimum_phi_t_ec_queue[i] = phi_t_ec_queue
            if phi_t_ec_queue < 0:
                if self._minimum_phi_t_ec_queue[i] == np.inf:
                    phi_t_4 += 1
                else:
                    phi_t_4 += phi_t_ec_queue / self._minimum_phi_t_ec_queue[i]
            elif phi_t_ec_queue > 0:
                if self._maximum_phi_t_ec_queue[i] == 0:
                    phi_t_4 += 1
                else:
                    phi_t_4 += phi_t_ec_queue / self._maximum_phi_t_ec_queue[i]
            else:
                phi_t_4 += 0
                
        phi_t_cc_queue = (self._cc_queue.get_queue(now) - self._cc_queue.get_output_by_time(now)) * self._cc_queue.get_input_by_time(now)
        if self._maximum_phi_t_cc_queue < phi_t_cc_queue:
            self._maximum_phi_t_cc_queue = phi_t_cc_queue
        if self._minimum_phi_t_cc_queue > phi_t_cc_queue:
            self._minimum_phi_t_cc_queue = phi_t_cc_queue
        if phi_t_cc_queue < 0:
            if self._minimum_phi_t_cc_queue == np.inf:
                phi_t_5 += 1
            else:
                phi_t_5 += phi_t_cc_queue / self._minimum_phi_t_cc_queue
        elif phi_t_cc_queue > 0:
            if self._maximum_phi_t_cc_queue == 0:
                phi_t_5 += 1
            else:
                phi_t_5 += phi_t_cc_queue / self._maximum_phi_t_cc_queue
        else:
            phi_t_5 += 0
        
        # print("phi_t_1: ", phi_t_1)
        # print("phi_t_2: ", phi_t_2)
        # print("phi_t_3: ", phi_t_3)
        # print("phi_t_4: ", phi_t_4)
        # print("phi_t_5: ", phi_t_5)    
        
        phi_t = 0.2 * (phi_t_1 / self._client_vehicle_num) + \
            0.2 * (phi_t_2 / self._client_vehicle_num) + \
            0.2 * (phi_t_3 / self._server_vehicle_num) + \
            0.2 * (phi_t_4 / self._edge_num) + \
            0.2 * (phi_t_5 / 1)
        
        return phi_t
    
    def compute_total_cost(
        self,
        task_offloading_actions: Dict, 
        transmission_power_allocation_actions: Dict,
        computation_resource_allocation_actions: Dict,
    ) -> float:  
        # init the costs
        v2v_transmission_costs = np.zeros((self._client_vehicle_num, ))
        v2i_transmission_costs = np.zeros((self._client_vehicle_num, ))
        lc_computing_costs = np.zeros((self._client_vehicle_num, ))
        
        vc_computing_costs = np.zeros((self._server_vehicle_num, ))
        
        ec_computing_costs = np.zeros((self._edge_num, ))
        i2i_transmission_costs = np.zeros((self._edge_num, ))
        i2c_transmission_costs = np.zeros((self._edge_num, ))
        
        cc_computing_cost = 0.0
        
        # compute the costs of the client vehicles
        for client_vehicle_index in range(self._client_vehicle_num):
            v2v_transmission_cost = compute_v2v_transmission_cost(
                client_vehicle_index=client_vehicle_index,
                client_vehicle_transmission_power=self._client_vehicles[client_vehicle_index].get_transmission_power(),
                transmission_power_allocation_actions=transmission_power_allocation_actions,
            )
            v2v_transmission_costs[client_vehicle_index] = v2v_transmission_cost
            if self._maximum_v2v_transmission_costs[client_vehicle_index] < v2v_transmission_costs[client_vehicle_index]:
                self._maximum_v2v_transmission_costs[client_vehicle_index] = v2v_transmission_costs[client_vehicle_index]
            
            v2i_transmission_costs[client_vehicle_index] = compute_v2i_transmission_cost(
                client_vehicle_index=client_vehicle_index,
                client_vehicle_transmission_power=self._client_vehicles[client_vehicle_index].get_transmission_power(),
                transmission_power_allocation_actions=transmission_power_allocation_actions,
            )
            if self._maximum_v2i_transmission_costs[client_vehicle_index] < v2i_transmission_costs[client_vehicle_index]:
                self._maximum_v2i_transmission_costs[client_vehicle_index] = v2i_transmission_costs[client_vehicle_index]
            
            lc_computing_costs[client_vehicle_index] = compute_lc_computing_cost(
                client_vehicle_index=client_vehicle_index,
                client_vehicle_computing_capability=self._client_vehicles[client_vehicle_index].get_computing_capability(),
                task_offloaded_at_client_vehicles=self._task_offloaded_at_client_vehicles,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            if self._maximum_lc_computing_costs[client_vehicle_index] < lc_computing_costs[client_vehicle_index]:
                self._maximum_lc_computing_costs[client_vehicle_index] = lc_computing_costs[client_vehicle_index]
        
        # compute the costs of the server vehicles
        for server_vehicle_index in range(self._server_vehicle_num):
            vc_computing_costs[server_vehicle_index] = compute_vc_computing_cost(
                server_vehicle_index=server_vehicle_index,
                server_vehicle_computing_capability=self._server_vehicles[server_vehicle_index].get_computing_capability(),
                task_offloaded_at_server_vehicles=self._task_offloaded_at_server_vehicles,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            if self._maximum_vc_computing_costs[server_vehicle_index] < vc_computing_costs[server_vehicle_index]:
                self._maximum_vc_computing_costs[server_vehicle_index] = vc_computing_costs[server_vehicle_index]
            
        # compute the costs of the edge nodes
        for edge_node_index in range(self._edge_num):
            ec_computing_costs[edge_node_index] = compute_ec_computing_cost(
                edge_node_index=edge_node_index,
                edge_node_computing_capability=self._edge_nodes[edge_node_index].get_computing_capability(),
                task_offloaded_at_edge_nodes=self._task_offloaded_at_edge_nodes,
                computation_resource_allocation_actions=computation_resource_allocation_actions,
            )
            if self._maximum_ec_computing_costs[edge_node_index] < ec_computing_costs[edge_node_index]:
                self._maximum_ec_computing_costs[edge_node_index] = ec_computing_costs[edge_node_index]
            
            i2i_transmission_costs[edge_node_index] = compute_i2i_transmission_cost(
                edge_node_index=edge_node_index,
                vehicles_under_V2I_communication_range=self._vehicles_under_V2I_communication_range,
                distance_matrix_between_edge_nodes=self._distance_matrix_between_edge_nodes,
                task_offloading_actions=task_offloading_actions, 
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                now=self.cur_step,
            )
            if self._maximum_i2i_transmission_costs[edge_node_index] < i2i_transmission_costs[edge_node_index]:
                self._maximum_i2i_transmission_costs[edge_node_index] = i2i_transmission_costs[edge_node_index]
            
            i2c_transmission_costs[edge_node_index] = compute_i2c_transmission_cost(
                edge_node_index=edge_node_index,
                vehicles_under_V2I_communication_range=self._vehicles_under_V2I_communication_range,
                distance_matrix_between_edge_nodes_and_the_cloud=self._distance_matrix_between_edge_nodes_and_the_cloud,
                task_offloading_actions=task_offloading_actions,
                client_vehicles=self._client_vehicles,
                client_vehicle_number=self._client_vehicle_num,
                maximum_task_generation_number=self._maximum_task_generation_number_of_vehicles,
                now=self.cur_step,
            )
            if self._maximum_i2c_transmission_costs[edge_node_index] < i2c_transmission_costs[edge_node_index]:
                self._maximum_i2c_transmission_costs[edge_node_index] = i2c_transmission_costs[edge_node_index]
            
        # compute the costs of the cloud
        cc_computing_cost = compute_cc_computing_cost(
            cloud_computing_capability=self._cloud.get_computing_capability(),
            task_offloaded_at_cloud=self._task_offloaded_at_cloud,
            computation_resource_allocation_actions=computation_resource_allocation_actions,
        )
        if self._maximum_cc_computing_cost < cc_computing_cost:
            self._maximum_cc_computing_cost = cc_computing_cost
        
        # print(
        #     'v2v_transmission_costs: ', v2v_transmission_costs,
        #     '\nv2i_transmission_costs: ', v2i_transmission_costs,
        #     '\nlc_computing_costs: ', lc_computing_costs,
        #     '\nvc_computing_costs: ', vc_computing_costs,
        #     '\nec_computing_costs: ', ec_computing_costs,
        #     '\ni2i_transmission_costs: ', i2i_transmission_costs,
        #     '\ni2c_transmission_costs: ', i2c_transmission_costs,
        #     '\ncc_computing_cost: ', cc_computing_cost,
        #     '\nmaximum_v2v_transmission_costs: ', self._maximum_v2v_transmission_costs,
        #     '\nmaximum_v2i_transmission_costs: ', self._maximum_v2i_transmission_costs,
        #     '\nmaximum_lc_computing_costs: ', self._maximum_lc_computing_costs,
        #     '\nmaximum_vc_computing_costs: ', self._maximum_vc_computing_costs,
        #     '\nmaximum_ec_computing_costs: ', self._maximum_ec_computing_costs,
        #     '\nmaximum_i2i_transmission_costs: ', self._maximum_i2i_transmission_costs,
        #     '\nmaximum_i2c_transmission_costs: ', self._maximum_i2c_transmission_costs,
        #     '\nmaximum_cc_computing_cost: ', self._maximum_cc_computing_cost,
        # )
        
        self._lc_costs[self.cur_step] = np.mean(lc_computing_costs)
        self._v2v_costs[self.cur_step] = np.mean(v2v_transmission_costs)
        self._v2i_costs[self.cur_step] = np.mean(v2i_transmission_costs)
        self._vc_costs[self.cur_step] = np.mean(vc_computing_costs)
        self._ec_costs[self.cur_step] = np.mean(ec_computing_costs)
        self._i2i_costs[self.cur_step] = np.mean(i2i_transmission_costs)
        self._i2c_costs[self.cur_step] = np.mean(i2c_transmission_costs)
        self._cc_costs[self.cur_step] = cc_computing_cost
        
        total_cost = compute_total_cost(
            client_vehicle_number=self._client_vehicle_num,
            v2v_transmission_costs=v2v_transmission_costs,
            maxmimum_v2v_transmission_costs=self._maximum_v2v_transmission_costs,
            v2i_transmission_costs=v2i_transmission_costs,
            maximum_v2i_transmission_costs=self._maximum_v2i_transmission_costs,
            lc_computing_costs=lc_computing_costs,
            maximum_lc_computing_costs=self._maximum_lc_computing_costs,
            server_vehicle_number=self._server_vehicle_num,
            vc_computing_costs=vc_computing_costs,
            maximum_vc_computing_costs=self._maximum_vc_computing_costs,
            edge_node_number=self._edge_num,
            ec_computing_costs=ec_computing_costs,
            maximum_ec_computing_costs=self._maximum_ec_computing_costs,
            i2i_transmission_costs=i2i_transmission_costs,
            maximum_i2i_transmission_costs=self._maximum_i2i_transmission_costs,
            i2c_transmission_costs=i2c_transmission_costs,
            maximum_i2c_transmission_costs=self._maximum_i2c_transmission_costs,
            cc_computing_cost=cc_computing_cost,
            maximum_cc_computing_cost=self._maximum_cc_computing_cost,
        )
        
        return total_cost
    
    def obtain_observation(self):
        observations = []
        
        observation = self.obser()
        
        for _ in range(self.n_agents):
            observations.append(observation)
        
        return observations
    
    def obser(self):
        # 初始化一个状态数组，形状为 (self._max_observation,)
        observation = np.zeros((self._max_observation, ))  # self.state_space 是通过 generate_state_space 生成的

        index = 0

        # 1. 客户端车辆的任务信息
        for client in range(self._client_vehicle_num):
            tasks_of_vehicle = self._client_vehicles[client].get_tasks_by_time(now=self.cur_step)  # 准备获取任务信息
            min_num = min(len(tasks_of_vehicle), self._maximum_task_generation_number_of_vehicles)
            for task in range(min_num):
                task_data_size = tasks_of_vehicle[task][2].get_input_data_size()
                CPU_cycles = tasks_of_vehicle[task][2].get_requested_computing_cycles()
                if self._maximum_task_data_size != 0:
                    observation[index] = task_data_size / self._maximum_task_data_size
                else:
                    observation[index] = 1
                index += 1
                if self._maximum_task_required_cycles != 0:
                    observation[index] = CPU_cycles / self._maximum_task_required_cycles
                else:
                    observation[index] = 1
                index += 1
            for _ in range(min_num, self._maximum_task_generation_number_of_vehicles):
                observation[index] = 0
                index += 1
                observation[index] = 0
                index += 1

        # 2. 客户端车辆的队列累积
        for client in range(self._client_vehicle_num):
            lc_backlog = self._lc_queue_backlogs[client][self.cur_step]  # 获取客户端车辆的队列累积
            observation[index] = lc_backlog
            if self._maximum_lc_queue_length[client] != 0:
                observation[index] /= self._maximum_lc_queue_length[client]
            else:
                observation[index] = 1
            index += 1

        # 3. V2V 连接信息
        v2v_connections = []
        for client in range(self._client_vehicle_num):
            for server in range(self._server_vehicle_num):
                v2v_connections.append(self._distance_matrix_between_client_vehicles_and_server_vehicles[client][server][self.cur_step])
        v2v_connections = np.array(v2v_connections).reshape(1, -1)
        v2v_embedded = self.v2v_pca.transform(v2v_connections)[0]
        observation[index:index + self.v2v_pca.n_components_] = v2v_embedded
        index += self.v2v_pca.n_components_
                
        # 4. 服务器车辆的队列累积
        for server in range(self._server_vehicle_num):
            v2v_backlog = self._v2v_queue_backlogs[server][self.cur_step]  # 获取服务器车辆的队列累积
            if self._maximum_v2v_queue_length[server] != 0:
                observation[index] = v2v_backlog / self._maximum_v2v_queue_length[server]
            else:
                observation[index] = 1
            index += 1
            vc_backlog = self._vc_queue_backlogs[server][self.cur_step]  # 获取服务器车辆的队列累积
            if self._maximum_vc_queue_length[server] != 0:
                observation[index] = vc_backlog / self._maximum_vc_queue_length[server]
            else:
                observation[index] = 1

            index += 1

        # 5. V2I 连接信息
        v2i_connections = []
        for client in range(self._client_vehicle_num):
            for edge in range(self._edge_num):
                v2i_connections.append(self._distance_matrix_between_client_vehicles_and_edge_nodes[client][edge][self.cur_step])
        v2i_connections = np.array(v2i_connections).reshape(1, -1)
        v2i_embedded = self.v2i_pca.transform(v2i_connections)[0]
        observation[index:index + self.v2i_pca.n_components_] = v2i_embedded
        index += self.v2i_pca.n_components_

        # 6. 边缘节点的队列累积
        for edge in range(self._edge_num):
            v2i_backlog = self._v2i_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            i2i_backlog = self._i2i_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            ec_backlog = self._ec_queue_backlogs[edge][self.cur_step]  # 获取边缘节点的队列累积
            if self._maximum_v2i_queue_length[edge] != 0:
                observation[index] = v2i_backlog / self._maximum_v2i_queue_length[edge]
            else:
                observation[index] = 1
            index += 1
            if self._maximum_i2i_queue_length[edge] != 0:
                observation[index] = i2i_backlog / self._maximum_i2i_queue_length[edge]
            else:
                observation[index] = 1
            index += 1
            if self._maximum_ec_queue_length[edge] != 0:
                observation[index] = ec_backlog / self._maximum_ec_queue_length[edge]
            else:
                observation[index] = 1
            index += 1

        # 7. 云的队列累积（I2C 和 CC）
        i2c_backlog = self._i2c_queue_backlogs[self.cur_step]  # 获取云的 I2C 队列
        if self._maximum_i2c_queue_length != 0:
            observation[index] = i2c_backlog / self._maximum_i2c_queue_length
        else:
            observation[index] = 1
        index += 1
        cc_backlog = self._cc_queue_backlogs[self.cur_step]  # 获取云的 CC 队列
        if self._maximum_cc_queue_length != 0:
            observation[index] = cc_backlog / self._maximum_cc_queue_length
        else:
            observation[index] = 1
        
        return observation

    def state(self):
        # 初始化一个状态数组，形状为 (state_space_number,)
        state = np.zeros(self.state_space.shape)  # self.state_space 是通过 generate_state_space 生成的

        state[0] = self.cur_step / (self._slot_length - 1)
        
        state[1] = 1 if self._done else 0
        
        return state

    def seed(self, seed):
        self._seed = seed

    def render(self):
        raise NotImplementedError

    def close(self):
        pass
    
    def process_actions(
        self,
        actions,
    ):
        processed_actions = []
        
        for i in range(self.n_agents):
            processed_action = np.zeros((self._max_action,))
            for j in range(self._max_action):
                # 将 actions[i][j] 从 [-1, 1] 范围映射到 [0, 1]
                normalized_action = (actions[i][j] + 1) / 2
                # 确保 normalized_action 在 [0, 1] 范围内
                normalized_action = max(0, min(1, normalized_action))  
                processed_action[j] = normalized_action
                
            processed_actions.append(processed_action)
        
        return processed_actions
        
    def transform_actions(
        self,
        actions,
        vehicles_under_V2V_communication_range,
    ):
        task_offloading_actions = {}
        transmission_power_allocation_actions = {}
        computation_resource_allocation_actions = {}
        
        for i in range(self.n_agents):
            true_action = actions[i][: self.true_action_space[i].shape[0]]
            if i == 0:      # task offloading agent
                task_offloading_number = 1 + self._maximum_server_vehicle_num + self._edge_num + 1
                for client_vehicle_index in range(self._client_vehicle_num):
                    tasks_of_vehicle = self._client_vehicles[client_vehicle_index].get_tasks_by_time(self.cur_step)
                    min_num = min(len(tasks_of_vehicle), self._maximum_task_generation_number_of_vehicles)
                    for j in range(min_num):
                        task_offloading_action = true_action[client_vehicle_index * self._maximum_task_generation_number_of_vehicles + j]
                        if task_offloading_action < 0 or task_offloading_action > 1:
                            raise ValueError("Task offloading action is out of range")
                        # transform the task offloading action to the task offloading index
                        # task_offloading_action [0, 1]
                        # 0: local, 1 - self._maximum_server_vehicle_num: server vehicle, self._maximum_server_vehicle_num + 1 - self._maximum_server_vehicle_num + self._edge_num: edge node, self._maximum_server_vehicle_num + self._edge_num + 1: cloud
                        task_offloading_index = math.floor(task_offloading_action * (task_offloading_number - 1))
                        if task_offloading_index == 0:
                            task_offloading_actions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(j)] = "Local"
                        elif task_offloading_index >= 1 and task_offloading_index <= self._maximum_server_vehicle_num:
                            tag = 0
                            flag = False
                            for server_vehicle_index in range(self._server_vehicle_num):
                                if vehicles_under_V2V_communication_range[client_vehicle_index][server_vehicle_index][self.cur_step] == 1:
                                    tag += 1
                                    if task_offloading_index == tag:
                                        flag = True
                                        task_offloading_actions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(j)] = "Server Vehicle " + str(server_vehicle_index)
                                        break
                            if not flag:
                                task_offloading_actions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(j)] = "Local"
                        elif task_offloading_index >= 1 + self._maximum_server_vehicle_num and task_offloading_index < 1 + self._maximum_server_vehicle_num + self._edge_num:
                            task_offloading_actions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(j)] = "Edge Node " + str(task_offloading_index - 1 - self._maximum_server_vehicle_num)
                        else:
                            task_offloading_actions["client_vehicle_" + str(client_vehicle_index) + "_task_" + str(j)] = "Cloud"
            elif i == 1:    # resource allocation agent
                client_action_number = self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2)
                server_action_number = self._server_vehicle_num * self._maximum_task_offloaded_at_server_vehicle_number
                edge_action_number = self._edge_num * self._maximum_task_offloaded_at_edge_node_number
                cloud_action_number = self._maximum_task_offloaded_at_cloud_number
                client_vehicle_actions = true_action[0: client_action_number]
                server_vehicle_actions = true_action[client_action_number: client_action_number + server_action_number]
                edge_node_actions = true_action[client_action_number + server_action_number: client_action_number + server_action_number + edge_action_number]
                cloud_actions = true_action[client_action_number + server_action_number + edge_action_number: client_action_number + server_action_number + edge_action_number + cloud_action_number]

                for i in range(self._client_vehicle_num):
                    client_vehicle_action = client_vehicle_actions[i * (self._maximum_task_offloaded_at_client_vehicle_number + 2): (i + 1) * (self._maximum_task_offloaded_at_client_vehicle_number + 2)]
                    computation_resource_allocation = client_vehicle_action[0: self._maximum_task_offloaded_at_client_vehicle_number]
                    transmission_power_allocation = client_vehicle_action[-2:]
                    computation_resource_allocation_normalized = self.normalize(computation_resource_allocation)
                    transmission_power_allocation_normalized = self.normalize(transmission_power_allocation)
                    computation_resource_allocation_actions["client_vehicle_" + str(i)] = computation_resource_allocation_normalized
                    transmission_power_allocation_actions["client_vehicle_" + str(i)] = transmission_power_allocation_normalized

                for i in range(self._server_vehicle_num):
                    computation_resource_allocation = server_vehicle_actions[i * self._maximum_task_offloaded_at_server_vehicle_number: (i + 1) * self._maximum_task_offloaded_at_server_vehicle_number]
                    computation_resource_allocation_normalized = self.normalize(computation_resource_allocation)
                    computation_resource_allocation_actions["server_vehicle_" + str(i)] = computation_resource_allocation_normalized

                for i in range(self._edge_num):
                    computation_resource_allocation = edge_node_actions[i * self._maximum_task_offloaded_at_edge_node_number: (i + 1) * self._maximum_task_offloaded_at_edge_node_number]
                    computation_resource_allocation_normalized = self.normalize(computation_resource_allocation)
                    computation_resource_allocation_actions["edge_node_" + str(i)] = computation_resource_allocation_normalized

                computation_resource_allocation = cloud_actions
                computation_resource_allocation_normalized = self.normalize(computation_resource_allocation)
                computation_resource_allocation_actions["cloud"] = computation_resource_allocation_normalized
            else:
                raise ValueError("Invalid agent index")
        return task_offloading_actions, transmission_power_allocation_actions, computation_resource_allocation_actions
        
    
    def obtain_tasks_offloading_conditions(
        self, 
        task_offloading_actions,
        vehicles_under_V2V_communication_range,
        vehicles_under_V2I_communication_range,
        now: int,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        task_offloaded_at_client_vehicles = {}
        task_offloaded_at_server_vehicles = {}
        task_offloaded_at_edge_nodes = {}
        task_uploaded_at_edge_nodes = {}
        task_offloaded_at_cloud = {}
        
        # init 
        for i in range(self._client_vehicle_num):
            task_offloaded_at_client_vehicles["client_vehicle_" + str(i)] = []
        for i in range(self._server_vehicle_num):
            task_offloaded_at_server_vehicles["server_vehicle_" + str(i)] = []
        for i in range(self._edge_num):
            task_offloaded_at_edge_nodes["edge_node_" + str(i)] = []
            task_uploaded_at_edge_nodes["edge_node_" + str(i)] = []
        task_offloaded_at_cloud["cloud"] = []
        
        for i in range(self._client_vehicle_num):
            tasks_of_vehicle_i = self._client_vehicles[i].get_tasks_by_time(now)
            min_num = min(len(tasks_of_vehicle_i), self._maximum_task_generation_number_of_vehicles)
            self._task_num_sum += min_num
            self._task_nums[self.cur_step] += min_num
            self._ave_task_num_of_each_client_vehicle[self.cur_step] += min_num
            if min_num > 0:
                for j in range(min_num):
                    task_key = f"client_vehicle_{i}_task_{j}"
                    # 确保任务键存在
                    if task_key not in task_offloading_actions:
                        continue
                    task_index = tasks_of_vehicle_i[j][1]
                    if task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)] == "Local":
                        if len(task_offloaded_at_client_vehicles["client_vehicle_" + str(i)]) < self._maximum_task_offloaded_at_client_vehicle_number:
                            task_offloaded_at_client_vehicles["client_vehicle_" + str(i)].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                            self._task_processed_at_local[self.cur_step] += 1
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Server Vehicle"):
                        server_vehicle_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        if vehicles_under_V2V_communication_range[i][server_vehicle_index][now] == 1:
                            if len(task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)]) < self._maximum_task_offloaded_at_server_vehicle_number:
                                task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)].append({"client_vehicle_index": i, "task_index": task_index, "task": tasks_of_vehicle_i[j][2]})
                                self._task_processed_at_vehicle[self.cur_step] += 1
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Edge Node"):
                        edge_node_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        for e in range(self._edge_num):
                            if vehicles_under_V2I_communication_range[i][e][now] == 1:
                                task_uploaded_at_edge_nodes["edge_node_" + str(e)].append({"client_vehicle_index": i, "task_index": task_index, "edge_index": e, "task": tasks_of_vehicle_i[j][2]})
                                self._task_uploaded_at_edge[self.cur_step] += 1
                                if len(task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)]) < self._maximum_task_offloaded_at_edge_node_number:
                                    task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)].append({"client_vehicle_index": i, "task_index": task_index, "edge_index": e, "task": tasks_of_vehicle_i[j][2]})
                                    self._task_processed_at_edge[self.cur_step] += 1
                                    if edge_node_index != e:
                                        self._task_processed_at_other_edge[self.cur_step] += 1
                                    else:
                                        self._task_processed_at_local_edge[self.cur_step] += 1
                                break
                    else:
                        for e in range(self._edge_num):
                            if vehicles_under_V2I_communication_range[i][e][now] == 1:
                                task_uploaded_at_edge_nodes["edge_node_" + str(e)].append({"client_vehicle_index": i, "task_index": task_index, "edge_index": e, "task": tasks_of_vehicle_i[j][2]})
                                self._task_uploaded_at_edge[self.cur_step] += 1
                                if len(task_offloaded_at_cloud["cloud"]) < self._maximum_task_offloaded_at_cloud_number:
                                    task_offloaded_at_cloud["cloud"].append({"client_vehicle_index": i, "task_index": task_index, "edge_index": e, "task": tasks_of_vehicle_i[j][2]})
                                    self._task_processed_at_cloud[self.cur_step] += 1
                                break
        
        ave_task_processed_at_local = 0
        ave_task_processed_at_vehicle = 0
        ave_task_processed_at_edge = 0
        ave_task_processed_at_cloud = 0
        avg_task_uploaded_at_each_edge = 0
        for i in range(self._client_vehicle_num):
            ave_task_processed_at_local += len(task_offloaded_at_client_vehicles["client_vehicle_" + str(i)])
        for i in range(self._server_vehicle_num):
            ave_task_processed_at_vehicle += len(task_offloaded_at_server_vehicles["server_vehicle_" + str(i)])
        for i in range(self._edge_num):
            ave_task_processed_at_edge += len(task_offloaded_at_edge_nodes["edge_node_" + str(i)])
            avg_task_uploaded_at_each_edge += len(task_uploaded_at_edge_nodes["edge_node_" + str(i)])
        ave_task_processed_at_cloud = len(task_offloaded_at_cloud["cloud"])
        ave_task_processed_at_local /= self._client_vehicle_num
        ave_task_processed_at_vehicle /= self._server_vehicle_num
        ave_task_processed_at_edge /= self._edge_num
        avg_task_uploaded_at_each_edge /= self._edge_num
        
        self._avg_task_processed_at_local[self.cur_step] = ave_task_processed_at_local
        self._avg_task_processed_at_vehicle[self.cur_step] = ave_task_processed_at_vehicle
        self._avg_task_processed_at_edge[self.cur_step] = ave_task_processed_at_edge
        self._avg_task_processed_at_cloud[self.cur_step] = ave_task_processed_at_cloud
        self._avg_task_uploaded_at_each_edge[self.cur_step] = avg_task_uploaded_at_each_edge
        
        self._task_successfully_processed_num[self.cur_step] = self._task_processed_at_local[self.cur_step] + self._task_processed_at_vehicle[self.cur_step] + self._task_processed_at_edge[self.cur_step] + self._task_processed_at_cloud[self.cur_step]
        self._ave_task_num_of_each_client_vehicle[self.cur_step] /= self._client_vehicle_num                       
        return task_offloaded_at_client_vehicles, task_offloaded_at_server_vehicles, task_offloaded_at_edge_nodes, task_uploaded_at_edge_nodes, task_offloaded_at_cloud
    
    def init_action_number_of_agents(self):

        # task offloading agent
        task_offloading_agent_action_number = self._client_vehicle_num * self._maximum_task_generation_number_of_vehicles

        # resource allocation agent
        resource_allocation_agent_action_number = self._client_vehicle_num * (self._maximum_task_offloaded_at_client_vehicle_number + 2) + \
                                                    self._server_vehicle_num * self._maximum_task_offloaded_at_server_vehicle_number + \
                                                    self._edge_num * self._maximum_task_offloaded_at_edge_node_number + \
                                                    self._maximum_task_offloaded_at_cloud_number
        
        max_action = max(task_offloading_agent_action_number, resource_allocation_agent_action_number)
        
        return max_action, task_offloading_agent_action_number, resource_allocation_agent_action_number
    
    def generate_action_space(self):
                
        action_space = []
        
        action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._max_action,), dtype=np.float32))
        action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._max_action,), dtype=np.float32))
        
        return action_space
    
    def generate_true_action_space(self):

        true_action_space = []
        
        true_action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._task_offloading_agent_action_number,), dtype=np.float32))
        true_action_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._resource_allocation_agent_action_number,), dtype=np.float32))
        
        return true_action_space
    
    def init_observation_number_of_agents(self):
        
        # the observation space of the task offloading agent
        # conisits of task information (data size, CPU cycles)
        # the queue backlog of the lc_queue
        # the V2V connection based on the V2V distance
        # the queue backlog of the V2V and VC queue of all server clients
        # the V2I connection based on the V2I distance
        # the queue backlog of the V2I, I2I, EC queue of all edge nodes
        # the queue backlog of the I2C and CC queue of the cloud
        task_offloading_agent_observation_number = \
            2 * self._maximum_task_generation_number_of_vehicles * self._client_vehicle_num + \
            self._client_vehicle_num + \
            self._v2v_n_components + \
            2 * self._server_vehicle_num + \
            self._v2i_n_components + \
            3 * self._edge_num + \
            2
        
        # the observation space of the resource allocation agent
        # conisits of task information (data size, CPU cycles)
        # the queue backlog of the lc_queue
        # the V2V connection based on the V2V distance
        # the queue backlog of the V2V and VC queue of all server clients
        # the V2I connection based on the V2I distance
        # the queue backlog of the V2I, I2I, EC queue of all edge nodes
        # the queue backlog of the I2C and CC queue of the cloud
    
        resource_allocation_agent_observation_number = \
            task_offloading_agent_observation_number
        
        max_observation = max(task_offloading_agent_observation_number, resource_allocation_agent_observation_number)
        
        return max_observation, task_offloading_agent_observation_number, resource_allocation_agent_observation_number
    
    def generate_observation_space(self):
        
        observation_space = []
        
        observation_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._task_offloading_agent_observation_number,), dtype=np.float32))
        observation_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._resource_allocation_agent_observation_number,), dtype=np.float32))
        
        return observation_space
    
    def generate_true_observation_space(self):
        
        true_observation_space = []
        
        true_observation_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._max_observation,), dtype=np.float32))
        true_observation_space.append(gym.spaces.Box(low=0.0, high=1.0, shape=(self._max_observation,), dtype=np.float32))
        
        return true_observation_space
    
    def generate_state_space(self):
        # the state space of the system
        # consists of time slot and is_done
        state_space_number = 2
        
        return gym.spaces.Box(low=0.0, high=1.0, shape=(state_space_number, ), dtype=np.float32)
    
    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
    
    def get_avail_actions(self):
        return None
    
    def normalize(self, x):
        total_sum = np.sum(x)
        if total_sum == 0:
            return np.zeros_like(x)
        return x / total_sum
    
    def get_env_tag(self):
        return self._env_tag

    def get_algorithm_type(self):
        return self._algorithm_type
    
    def save_results(
        self,
        env_tag: str = "",
        alg_type: str = "",
    ):
        """add results to the file"""
        try:
            env_tag = env_tag if env_tag != "" else self.get_env_tag()
            alg_type = alg_type if alg_type != "" else self.get_algorithm_type()
            sample_file_name = self._resutls_file_name + "_" + env_tag + "_" + alg_type + ".txt"
            complete_file_name = self._resutls_file_name + "_" + env_tag + "_" + alg_type + "_complete.txt"
            with open(sample_file_name, "a+") as f:
                f.write("episode_index: " + str(self._episode_index) + "\n" \
                    + "reward: " + str(self._reward) + "\n" \
                    + "rewards: " + str(self._rewards) + "\n" \
                    + "average reward: " + str(np.mean(self._rewards)) + "\n" \
                    + "costs: " + str(self._costs) + "\n" \
                    + "average cost: " + str(np.mean(self._costs)) + "\n" \
                    + "average lc cost: " + str(np.mean(self._lc_costs)) + "\n" \
                    + "average v2v cost: " + str(np.mean(self._v2v_costs)) + "\n" \
                    + "average vc cost: " + str(np.mean(self._vc_costs)) + "\n" \
                    + "average v2i cost: " + str(np.mean(self._v2i_costs)) + "\n" \
                    + "average i2i cost: " + str(np.mean(self._i2i_costs)) + "\n" \
                    + "average ec cost: " + str(np.mean(self._ec_costs)) + "\n" \
                    + "average i2c cost: " + str(np.mean(self._i2c_costs)) + "\n" \
                    + "average cc cost: " + str(np.mean(self._cc_costs)) + "\n" \
                    + "average delay queue: " + str(np.mean(self._total_delay_queues)) + "\n" \
                    + "average lc delay queue: " + str(np.mean(self._lc_delay_queues)) + "\n" \
                    + "average v2v delay queue: " + str(np.mean(self._v2v_delay_queues)) + "\n" \
                    + "average vc delay queue: " + str(np.mean(self._vc_delay_queues)) + "\n" \
                    + "average v2i delay queue: " + str(np.mean(self._v2i_delay_queues)) + "\n" \
                    + "average i2i delay queue: " + str(np.mean(self._i2i_delay_queues)) + "\n" \
                    + "average ec delay queue: " + str(np.mean(self._ec_delay_queues)) + "\n" \
                    + "average i2c delay queue: " + str(np.mean(self._i2c_delay_queues)) + "\n" \
                    + "average cc delay queue: " + str(np.mean(self._cc_delay_queues)) + "\n" \
                    + "average lc res queue: " + str(np.mean(self._lc_res_queues)) + "\n" \
                    + "average vc res queue: " + str(np.mean(self._vc_res_queues)) + "\n" \
                    + "average ec res queue: " + str(np.mean(self._ec_res_queues)) + "\n" \
                    + "average cc res queue: " + str(np.mean(self._cc_res_queues)) + "\n" \
                    + "average task num: " + str(self._task_num_sum / self._slot_length) + "\n" \
                    + "average task uploaded at edge: " + str(np.mean(self._task_uploaded_at_edge)) + "\n" \
                    + "average task processed at local: " + str(np.mean(self._task_processed_at_local)) + "\n" \
                    + "average task processed at vehicle: " + str(np.mean(self._task_processed_at_vehicle)) + "\n" \
                    + "average task processed at edge: " + str(np.mean(self._task_processed_at_edge)) + "\n" \
                    + "average task processed at local edge: " + str(np.mean(self._task_processed_at_local_edge)) + "\n" \
                    + "average task processed at other edge: " + str(np.mean(self._task_processed_at_other_edge)) + "\n" \
                    + "average task processed at cloud: " + str(np.mean(self._task_processed_at_cloud)) + "\n" \
                    + "average task successfully processed num: " + str(np.mean(self._task_successfully_processed_num)) + "\n" \
                    + "average avg task num of each client vehicle: " + str(np.mean(self._ave_task_num_of_each_client_vehicle)) + "\n" \
                    + "average avg task processed at local: " + str(np.mean(self._avg_task_processed_at_local)) + "\n" \
                    + "average avg task processed at vehicle: " + str(np.mean(self._avg_task_processed_at_vehicle)) + "\n" \
                    + "average avg task processed at edge: " + str(np.mean(self._avg_task_processed_at_edge)) + "\n" \
                    + "average avg task processed at cloud: " + str(np.mean(self._avg_task_processed_at_cloud)) + "\n" \
                    + "average avg task uploaded at each edge: " + str(np.mean(self._avg_task_uploaded_at_each_edge)) + "\n" \
                    + "*************************************************************************************************************" + "\n")
            # with open(complete_file_name, "a+") as f:
            #     f.write("episode_index: " + str(self._episode_index) + "\n" \
            #         + "reward: " + str(self._reward) + "\n" \
            #         + "rewards: " + str(self._rewards) + "\n" \
            #         + "average reward: " + str(np.mean(self._rewards)) + "\n" \
            #         + "costs: " + str(self._costs) + "\n" \
            #         + "average cost: " + str(np.mean(self._costs)) + "\n" \
            #         + "lc_costs: " + str(self._lc_costs) + "\n" \
            #         + "average lc cost: " + str(np.mean(self._lc_costs)) + "\n" \
            #         + "v2v_costs: " + str(self._v2v_costs) + "\n" \
            #         + "average v2v cost: " + str(np.mean(self._v2v_costs)) + "\n" \
            #         + "vc_costs: " + str(self._vc_costs) + "\n" \
            #         + "average vc cost: " + str(np.mean(self._vc_costs)) + "\n" \
            #         + "v2i_costs: " + str(self._v2i_costs) + "\n" \
            #         + "average v2i cost: " + str(np.mean(self._v2i_costs)) + "\n" \
            #         + "i2i_costs: " + str(self._i2i_costs) + "\n" \
            #         + "average i2i cost: " + str(np.mean(self._i2i_costs)) + "\n" \
            #         + "ec_costs: " + str(self._ec_costs) + "\n" \
            #         + "average ec cost: " + str(np.mean(self._ec_costs)) + "\n" \
            #         + "i2c_costs: " + str(self._i2c_costs) + "\n" \
            #         + "average i2c cost: " + str(np.mean(self._i2c_costs)) + "\n" \
            #         + "cc_costs: " + str(self._cc_costs) + "\n" \
            #         + "average cc cost: " + str(np.mean(self._cc_costs)) + "\n" \
            #         + "delay_queues: " + str(self._total_delay_queues) + "\n" \
            #         + "average delay queue: " + str(np.mean(self._total_delay_queues)) + "\n" \
            #         + "lc_delay_queues: " + str(self._lc_delay_queues) + "\n" \
            #         + "average lc delay queue: " + str(np.mean(self._lc_delay_queues)) + "\n" \
            #         + "v2v_delay_queues: " + str(self._v2v_delay_queues) + "\n" \
            #         + "average v2v delay queue: " + str(np.mean(self._v2v_delay_queues)) + "\n" \
            #         + "vc_delay_queues: " + str(self._vc_delay_queues) + "\n" \
            #         + "average vc delay queue: " + str(np.mean(self._vc_delay_queues)) + "\n" \
            #         + "v2i_delay_queues: " + str(self._v2i_delay_queues) + "\n" \
            #         + "average v2i delay queue: " + str(np.mean(self._v2i_delay_queues)) + "\n" \
            #         + "i2i_delay_queues: " + str(self._i2i_delay_queues) + "\n" \
            #         + "average i2i delay queue: " + str(np.mean(self._i2i_delay_queues)) + "\n" \
            #         + "ec_delay_queues: " + str(self._ec_delay_queues) + "\n" \
            #         + "average ec delay queue: " + str(np.mean(self._ec_delay_queues)) + "\n" \
            #         + "i2c_delay_queues: " + str(self._i2c_delay_queues) + "\n" \
            #         + "average i2c delay queue: " + str(np.mean(self._i2c_delay_queues)) + "\n" \
            #         + "cc_delay_queues: " + str(self._cc_delay_queues) + "\n" \
            #         + "average cc delay queue: " + str(np.mean(self._cc_delay_queues)) + "\n" \
            #         + "lc_res_queues: " + str(self._lc_res_queues) + "\n" \
            #         + "average lc res queue: " + str(np.mean(self._lc_res_queues)) + "\n" \
            #         + "vc_res_queues: " + str(self._vc_res_queues) + "\n" \
            #         + "average vc res queue: " + str(np.mean(self._vc_res_queues)) + "\n" \
            #         + "ec_res_queues: " + str(self._ec_res_queues) + "\n" \
            #         + "average ec res queue: " + str(np.mean(self._ec_res_queues)) + "\n" \
            #         + "cc_res_queues: " + str(self._cc_res_queues) + "\n" \
            #         + "average cc res queue: " + str(np.mean(self._cc_res_queues)) + "\n" \
            #         + "task_num_sum: " + str(self._task_num_sum) + "\n" \
            #         + "average task num: " + str(self._task_num_sum / self._slot_length) + "\n" \
            #         + "task_nums: " + str(self._task_nums) + "\n" \
            #         + "task_uploaded_at_edge: " + str(self._task_uploaded_at_edge) + "\n" \
            #         + "average task uploaded at edge: " + str(np.mean(self._task_uploaded_at_edge)) + "\n" \
            #         + "task_processed_at_local: " + str(self._task_processed_at_local) + "\n" \
            #         + "average task processed at local: " + str(np.mean(self._task_processed_at_local)) + "\n" \
            #         + "task_processed_at_vehicle: " + str(self._task_processed_at_vehicle) + "\n" \
            #         + "average task processed at vehicle: " + str(np.mean(self._task_processed_at_vehicle)) + "\n" \
            #         + "task_processed_at_edge: " + str(self._task_processed_at_edge) + "\n" \
            #         + "average task processed at edge: " + str(np.mean(self._task_processed_at_edge)) + "\n" \
            #         + "task_processed_at_local_edge: " + str(self._task_processed_at_local_edge) + "\n" \
            #         + "average task processed at local edge: " + str(np.mean(self._task_processed_at_local_edge)) + "\n" \
            #         + "task_processed_at_other_edge: " + str(self._task_processed_at_other_edge) + "\n" \
            #         + "average task processed at other edge: " + str(np.mean(self._task_processed_at_other_edge)) + "\n" \
            #         + "task_processed_at_cloud: " + str(self._task_processed_at_cloud) + "\n" \
            #         + "average task processed at cloud: " + str(np.mean(self._task_processed_at_cloud)) + "\n" \
            #         + "task_successfully_processed_num: " + str(self._task_successfully_processed_num) + "\n" \
            #         + "average task successfully processed num: " + str(np.mean(self._task_successfully_processed_num)) + "\n" \
            #         + "avg task num of each client vehicle: " + str(self._ave_task_num_of_each_client_vehicle) + "\n" \
            #         + "average avg task num of each client vehicle: " + str(np.mean(self._ave_task_num_of_each_client_vehicle)) + "\n" \
            #         + "avg task processed at local: " + str(self._avg_task_processed_at_local) + "\n" \
            #         + "average avg task processed at local: " + str(np.mean(self._avg_task_processed_at_local)) + "\n" \
            #         + "avg task processed at vehicle: " + str(self._avg_task_processed_at_vehicle) + "\n" \
            #         + "average avg task processed at vehicle: " + str(np.mean(self._avg_task_processed_at_vehicle)) + "\n" \
            #         + "avg task processed at edge: " + str(self._avg_task_processed_at_edge) + "\n" \
            #         + "average avg task processed at edge: " + str(np.mean(self._avg_task_processed_at_edge)) + "\n" \
            #         + "avg task processed at cloud: " + str(self._avg_task_processed_at_cloud) + "\n" \
            #         + "average avg task processed at cloud: " + str(np.mean(self._avg_task_processed_at_cloud)) + "\n" \
            #         + "avg task uploaded at each edge: " + str(self._avg_task_uploaded_at_each_edge) + "\n" \
            #         + "average avg task uploaded at each edge: " + str(np.mean(self._avg_task_uploaded_at_each_edge)) + "\n" \
            #         + "*************************************************************************************************************" + "\n")
        except:
            raise Exception("No such file: " + self._resutls_file_name)