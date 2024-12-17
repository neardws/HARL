"""Train an algorithm."""
from harl.runners import RUNNER_REGISTRY

if __name__ == "__main__":
    # start training
    algo = "happo"
    args = {
        'algo': 'happo',
        'env': 'pettingzoo_mpe',
        'exp_name': 'installtest',
        'load_config': ''
    }
    algo_args = {
        'seed': {
            'seed_specify': True,
            'seed': 1
        },
        'device': {
            'cuda': True,
            'cuda_deterministic': True,
            'torch_threads': 4
        },
        'train': {
            'n_rollout_threads': 20,
            'num_env_steps': 10000000,
            'episode_length': 200,
            'log_interval': 5,
            'eval_interval': 25,
            'use_valuenorm': True,
            'use_linear_lr_decay': False,
            'use_proper_time_limits': True,
            'model_dir': None
        },
        'eval': {
            'use_eval': True,
            'n_eval_rollout_threads': 10,
            'eval_episodes': 20
        },
        'render': {
            'use_render': False,
            'render_episodes': 10
        },
        'model': {
            'hidden_sizes': [128, 128],
            'activation_func': 'relu',
            'use_feature_normalization': True,
            'initialization_method': 'orthogonal_',
            'gain': 0.01,
            'use_naive_recurrent_policy': False,
            'use_recurrent_policy': False,
            'recurrent_n': 1,
            'data_chunk_length': 10,
            'lr': 0.0005,
            'critic_lr': 0.0005,
            'opti_eps': 1e-05,
            'weight_decay': 0,
            'std_x_coef': 1,
            'std_y_coef': 0.5
        },
        'algo': {
            'ppo_epoch': 5,
            'critic_epoch': 5,
            'use_clipped_value_loss': True,
            'clip_param': 0.2,
            'actor_num_mini_batch': 1,
            'critic_num_mini_batch': 1,
            'entropy_coef': 0.01,
            'value_loss_coef': 1,
            'use_max_grad_norm': True,
            'max_grad_norm': 10.0,
            'use_gae': True,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'use_huber_loss': True,
            'use_policy_active_masks': True,
            'huber_delta': 10.0,
            'action_aggregation': 'prod',
            'share_param': False,
            'fixed_order': False
        },
        'logger': {
            'log_dir': './results'
        }
    }
    env_args = {'scenario': 'simple_spread_v2', 'continuous_actions': True}

    runner = RUNNER_REGISTRY[algo](args, algo_args, env_args)
    runner.run()
    runner.close()
