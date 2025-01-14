from harl.envs.vec.vec_env import VECEnv
from other_algorithms.greedy_agent import GreedyAgent
import yaml
import os

if __name__ == '__main__':
    
    step_nums = 50 * 2500
    
    # Scenario 1
    env_cfg_path = "/root/HARL/env_configs/scenairo/vec_1_70_05.yaml"
    # # Scenario 2
    # env_cfg_path = "/root/HARL/env_configs/scenairo/vec_2_70_05.yaml"
    # # Scenario 3
    # env_cfg_path = "/root/HARL/env_configs/scenairo/vec_3_70_05.yaml"

    # # Arrival Rate 0.3
    # env_cfg_path = "/root/HARL/env_configs/arrival_rate/vec_1_70_03.yaml"
    # # Arrival Rate 0.4
    # env_cfg_path = "/root/HARL/env_configs/arrival_rate/vec_1_70_04.yaml"
    # # Arrival Rate 0.6
    # env_cfg_path = "/root/HARL/env_configs/arrival_rate/vec_1_70_06.yaml"
    # # Arrival Rate 0.7
    # env_cfg_path = "/root/HARL/env_configs/arrival_rate/vec_1_70_07.yaml"

    # # Vehicle Number 50
    # env_cfg_path = "/root/HARL/env_configs/vehicle_number/vec_1_50_05.yaml"
    # # Vehicle Number 60
    # env_cfg_path = "/root/HARL/env_configs/vehicle_number/vec_1_60_05.yaml"
    # # Vehicle Number 80
    # env_cfg_path = "/root/HARL/env_configs/vehicle_number/vec_1_80_05.yaml"
    # # Vehicle Number 90
    # env_cfg_path = "/root/HARL/env_configs/vehicle_number/vec_1_90_05.yaml"
    
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)

    env = VECEnv(env_args)
    
    agent = GreedyAgent(
        client_vehicle_num = env.get_client_vehicle_num(),
        maximum_task_generation_number = env.get_maximum_task_generation_number_of_vehicles(),
        maximum_server_vehicle_num = env.get_maximum_server_vehicle_num(),
        server_vehicle_num = env.get_server_vehicle_num(),
        edge_num = env.get_edge_num(),
        max_action = env.get_max_action(),
        agent_number = env.get_n_agents(),
        maximum_task_offloaded_at_client_vehicle_number = env.get_maximum_task_offloaded_at_client_vehicle_number(),
        maximum_task_offloaded_at_server_vehicle_number = env.get_maximum_task_offloaded_at_server_vehicle_number(),
        maximum_task_offloaded_at_edge_node_number = env.get_maximum_task_offloaded_at_edge_node_number(),
        maximum_task_offloaded_at_cloud_number = env.get_maximum_task_offloaded_at_cloud_number(),
        v2v_n_components = env.get_v2v_n_components(),
        v2i_n_components = env.get_v2i_n_components(),
    )
    
    max_action = env.get_max_action()
    n_agents = env.get_n_agents()
    
    for _ in range(step_nums):
        obs = env.obtain_observation()
        v2v_connections = env.get_v2v_connections()
        v2i_connections = env.get_v2i_connections()
        env.step(
            actions = agent.generate_action(obs = obs, v2v_connections = v2v_connections, v2i_connections = v2i_connections)) 
    
    os.system("/usr/bin/shutdown")