"""Train an algorithm."""
from harl.runners import RUNNER_REGISTRY
from harl.utils.configs_tools import get_defaults_yaml_args

if __name__ == "__main__":
    # start training
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'vec',
        'exp_name': 'test',
        'load_config': ''
    }
    algo_args, env_args = get_defaults_yaml_args(algo, args['env'])

    runner = RUNNER_REGISTRY[algo](args, algo_args, env_args)
    runner.run()
    runner.close()
