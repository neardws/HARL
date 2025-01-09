from harl.envs.vec.vec_env import VECEnv
from other_algorithms.randomized_rounding import randomized_rounding_agent
from harl.utils.configs_tools import get_defaults_yaml_args

if __name__ == '__main__':
    
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'vec',
        'exp_name': 'test_randomized_rounding',
        'load_config': ''
    }
    algo_args, env_args = get_defaults_yaml_args(algo, args['env'])
    
    env = VECEnv(env_args)
    
    agent = randomized_rounding_agent(
        client_vehicle_num = env.get_client_vehicle_num(),
        maximum_task_generation_number_of_vehicles = env.get_maximum_task_generation_number_of_vehicles(),
        maximum_server_vehicle_num = env.get_maximum_server_vehicle_num(),
        edge_num = env.get_edge_num(),
        max_action = env.get_max_action(),
        agent_number = env.get_n_agents()
    )
    
    max_action = env.get_max_action()
    n_agents = env.get_n_agents()
    
    for _ in range(60 * 1000):
        env.step(actions = agent.generate_action()) 