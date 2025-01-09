from harl.envs.vec.vec_env import VECEnv
from other_algorithms.greedy_agent import GreedyAgent
from harl.utils.configs_tools import get_defaults_yaml_args

if __name__ == '__main__':
    
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'vec',
        'exp_name': 'test_greedy',
        'load_config': ''
    }
    algo_args, env_args = get_defaults_yaml_args(algo, args['env'])
    
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
    
    for _ in range(60 * 1000):
        obs = env.obtain_observation()
        v2v_connections = env.get_v2v_connections()
        v2i_connections = env.get_v2i_connections()
        env.step(actions = agent.generate_action(obs = obs, v2v_connections = v2v_connections, v2i_connections = v2i_connections)) 