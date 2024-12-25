from harl.envs.vec.vec_env import VECEnv 
from other_algorithms.random_actions import ra_agent
from harl.utils.configs_tools import get_defaults_yaml_args

if __name__ == '__main__':   
    
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'vec',
        'exp_name': 'test',
        'load_config': ''
    }
    algo_args, env_args = get_defaults_yaml_args(algo, args['env'])
    
    env = VECEnv(env_args)
    agent = ra_agent()
    
    max_action = env.get_max_action()
    n_agents = env.get_n_agents()    
    
    for _ in range(120):
        if env.is_done():
            env.reset()
        env.step(
            actions = agent.generate_action(
                max_action = max_action,
                agent_number = n_agents
            )
        )

    