"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY


    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    
    print("args algo", args["algo"])
    print("args", args)
    print("algo_args", algo_args)
    print("env_args", env_args)
    
    # args algo happo
    # args {'algo': 'happo', 'env': 'pettingzoo_mpe', 'exp_name': 'installtest', 'load_config': ''}
    # algo_args {'seed': {'seed_specify': True, 'seed': 1}, 'device': {'cuda': True, 'cuda_deterministic': True, 'torch_threads': 4}, 'train': {'n_rollout_threads': 20, 'num_env_steps': 10000000, 'episode_length': 200, 'log_interval': 5, 'eval_interval': 25, 'use_valuenorm': True, 'use_linear_lr_decay': False, 'use_proper_time_limits': True, 'model_dir': None}, 'eval': {'use_eval': True, 'n_eval_rollout_threads': 10, 'eval_episodes': 20}, 'render': {'use_render': False, 'render_episodes': 10}, 'model': {'hidden_sizes': [128, 128], 'activation_func': 'relu', 'use_feature_normalization': True, 'initialization_method': 'orthogonal_', 'gain': 0.01, 'use_naive_recurrent_policy': False, 'use_recurrent_policy': False, 'recurrent_n': 1, 'data_chunk_length': 10, 'lr': 0.0005, 'critic_lr': 0.0005, 'opti_eps': 1e-05, 'weight_decay': 0, 'std_x_coef': 1, 'std_y_coef': 0.5}, 'algo': {'ppo_epoch': 5, 'critic_epoch': 5, 'use_clipped_value_loss': True, 'clip_param': 0.2, 'actor_num_mini_batch': 1, 'critic_num_mini_batch': 1, 'entropy_coef': 0.01, 'value_loss_coef': 1, 'use_max_grad_norm': True, 'max_grad_norm': 10.0, 'use_gae': True, 'gamma': 0.99, 'gae_lambda': 0.95, 'use_huber_loss': True, 'use_policy_active_masks': True, 'huber_delta': 10.0, 'action_aggregation': 'prod', 'share_param': False, 'fixed_order': False}, 'logger': {'log_dir': './results'}}
    # env_args {'scenario': 'simple_spread_v2', 'continuous_actions': True}
    
    
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
