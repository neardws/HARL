"""Train an algorithm."""
from harl.runners import RUNNER_REGISTRY
import yaml
import os

if __name__ == "__main__":

    algo = "maddpg"

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

    algo_cfg_path = "/root/HARL/harl/configs/algos_cfgs/maddpg.yaml"

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    
    args = {
        'algo': 'maddpg',
        'env': 'vec',
        'exp_name': 'scenario_1',
        'load_config': ''
    }

    runner = RUNNER_REGISTRY[algo](args, algo_args, env_args)
    runner.run()
    runner.close()

    os.system("/usr/bin/shutdown")