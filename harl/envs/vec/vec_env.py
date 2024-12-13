import gym
import numpy as np
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
from utilities.time_calculation import obtain_computing_time, obtain_transmission_time, obtain_wired_transmission_time
from utilities.time_calculation import compute_transmission_rate, compute_V2V_SINR, compute_V2I_SINR, compute_INR, compute_S
from utilities.conversion import cover_mW_to_W

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
        self.share_observation_space = ...
        self.observation_space = ...
        self.action_space = self.generate_action_space()
        
        self.value_normalizer = ValueNorm(1, device=self.device)
        
        
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
        
        self._lc_queue_backlogs = np.zeros((self._client_vehicle_num, ))
        self._v2v_queue_backlogs = np.zeros((self._server_vehicle_num, ))
        self._vc_queue_backlogs = np.zeros((self._server_vehicle_num, ))
        self._v2i_queue_backlogs = np.zeros((self._edge_node_number, ))
        self._i2i_queue_backlogs = np.zeros((self._edge_node_number, ))
        self._ec_queue_backlogs = np.zeros((self._edge_node_number, ))
        self._i2c_queue_backlogs = np.zeros((1, ))
        self._cc_queue_backlogs = np.zeros((1, ))
        

    def step(self, actions):
        pass
        # return obs, state, rewards, dones, info, available_actions

    def reset(self):
        pass
        # return obs, state, available_actions

    def seed(self, seed):
        self._seed = seed

    def render(self):
        raise NotImplementedError

    def close(self):
        self.env.close()
        
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
    
    def generate_action_space(self):
        if self._n_agents != self._maximum_client_vehicle_number * 2 + self._edge_node_number + 1:
            raise ValueError("The number of agents is not correct.")
        
        task_offloading_number = 1 + self._maximum_server_vehicle_number + self._maximum_server_vehicle_number + self._edge_node_number + 1
        
        action_space = []
        for i in range(self._n_agents):
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
                            task_offloaded_at_client_vehicles["client_vehicle_" + str(i)].append({"client_vehicle_index": i, "task_index": task_index})
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Server Vehicle"):
                        server_vehicle_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        if vehicles_under_V2V_communication_range[i][server_vehicle_index] == 1:
                            if len(task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)]) < self._maximum_task_offloaded_at_server_vehicle_number:
                                task_offloaded_at_server_vehicles["server_vehicle_" + str(server_vehicle_index)].append({"client_vehicle_index": i, "task_index": task_index})
                    elif task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].startswith("Edge Node"):
                        edge_node_index = int(task_offloading_actions["client_vehicle_" + str(i) + "_task_" + str(j)].split(" ")[-1])
                        if any(vehicles_under_V2I_communication_range[i][e] == 1 for e in range(self._edge_node_number)):
                            if len(task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)]) < self._maximum_task_offloaded_at_edge_node_number:
                                task_offloaded_at_edge_nodes["edge_node_" + str(edge_node_index)].append({"client_vehicle_index": i, "task_index": task_index})
                    else:
                        if any(vehicles_under_V2I_communication_range[i][e] == 1 for e in range(self._edge_node_number)):
                            if len(task_offloaded_at_cloud["cloud"]) < self._maximum_task_offloaded_at_cloud_number:
                                task_offloaded_at_cloud["cloud"].append({"client_vehicle_index": i, "task_index": task_index})
                                
        return task_offloaded_at_client_vehicles, task_offloaded_at_server_vehicles, task_offloaded_at_edge_nodes, task_offloaded_at_cloud
    
    def generate_observation_space(self):
        pass
    
    def generate_share_observation_space(self):
        pass
    
    def generate_state_space(self):
        pass
    
    def generate_reward_space(self):
        pass
    
    def generate_avail_actions(self):
        pass
    