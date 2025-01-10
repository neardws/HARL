from harl.envs.vec.vec_env import VECEnv
from other_algorithms.dp_agent import DPAgent
from harl.utils.configs_tools import get_defaults_yaml_args

if __name__ == '__main__':
    
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'vec',
        'exp_name': 'test_dp',
        'load_config': ''
    }
    algo_args, env_args = get_defaults_yaml_args(algo, args['env'])
    
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
    for step in range(60 * 1000):
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
