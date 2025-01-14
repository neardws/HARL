from harl.envs.vec.vec_env import VECEnv
from other_algorithms.dp_agent import DPAgent
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
    
    agent = DPAgent(
        client_vehicle_num = env.get_client_vehicle_num(),
        maximum_task_generation_number_of_vehicles = env.get_maximum_task_generation_number_of_vehicles(),
        maximum_server_vehicle_num = env.get_maximum_server_vehicle_num(),
        server_vehicle_num = env.get_server_vehicle_num(),
        edge_num = env.get_edge_num(),
        max_action = env.get_max_action(),
        maximum_task_offloaded_at_client_vehicle_number = env.get_maximum_task_offloaded_at_client_vehicle_number(),
        maximum_task_offloaded_at_server_vehicle_number = env.get_maximum_task_offloaded_at_server_vehicle_number(),
        maximum_task_offloaded_at_edge_node_number = env.get_maximum_task_offloaded_at_edge_node_number(),
        maximum_task_offloaded_at_cloud_number = env.get_maximum_task_offloaded_at_cloud_number(),
        agent_number = env.get_n_agents(),
        state_horizon = 3,
        discount_factor = 0.9,
        threshold = 1e-4,
        use_dp = True,  # 启用动态规划
        warmup_steps = 1000  # 设置预热步数
    )
    
    print("Starting environment loop...")
    for step in range(step_nums):
        obs = env.obtain_observation()
        
        current_state = agent._get_state_key(obs[0])
        
        actions = agent.generate_action(obs=obs)

        # 获取环境信息
        env_info = {
            'reward': env.get_reward(),
            'next_obs': obs
        }
        agent.store_transition(obs=obs, current_state=current_state, actions=actions, env_info=env_info)
        env.step(actions=actions) 


    os.system("/usr/bin/shutdown")