import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import torch
from arguments_rl import get_args

from collections import deque
from rl_module.a2c_ppo_acktr.envs import make_vec_envs
from rl_module.a2c_ppo_acktr.storage import RolloutStorage
from rl_module.train_ppo import train_ppo

def main():

    # Arguments
    args = get_args()
    conv_experiment = [
        'atari',
    ]

    # Split
    ##########################################################################################################################33
    if args.approach == 'fine-tuning' or args.approach == 'ft-fix':
        log_name = '{}_{}_{}_{}'.format(args.date, args.experiment, args.approach,args.seed)
    elif args.approach == 'ewc' in args.approach:
        log_name = '{}_{}_{}_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, args.ewc_lambda)
    elif args.approach == 'blip':
        log_name = '{}_{}_{}_{}_F_prior_{}'.format(args.date, args.experiment, args.approach, args.seed, args.F_prior)

    if args.experiment in conv_experiment:
        log_name = log_name + '_conv'

    ########################################################################################################################
    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print('[CUDA unavailable]'); sys.exit()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Inits
    print('Inits...')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda:0" if args.cuda else "cpu")

    taskcla = [(0,14), (1,18), (2,18), (3,18), (4,18), (5,6)]
    task_sequences = [(0, 'KungFuMasterNoFrameskip-v4'), (1, 'BoxingNoFrameskip-v4'),
                    (2, 'JamesbondNoFrameskip-v4'), (3, 'KrullNoFrameskip-v4'),
                    (4, 'RiverraidNoFrameskip-v4'), (5, 'SpaceInvadersNoFrameskip-v4')]

    # hard coded for atari environment
    obs_shape = (4,84,84)

    if args.approach == 'blip':
        from rl_module.ppo_model import QPolicy
        print('using fisher prior of: ', args.F_prior)
        actor_critic = QPolicy(obs_shape,
            taskcla,
            base_kwargs={'F_prior': args.F_prior, 'recurrent': args.recurrent_policy}).to(device)
    else:
        from rl_module.ppo_model import Policy
        actor_critic = Policy(obs_shape,
            taskcla,
            base_kwargs={'recurrent': args.recurrent_policy}).to(device)

    # Args -- Approach
    if args.approach == 'fine-tuning' or args.approach == 'ft-fix':
        from rl_module.ppo import PPO as approach

        agent = approach(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                use_clipped_value_loss=True)
    elif args.approach == 'ewc':
        from rl_module.ppo_ewc import PPO_EWC as approach

        agent = approach(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            ewc_lambda= args.ewc_lambda,
            online = args.ewc_online)

    elif args.approach == 'blip':
        from rl_module.ppo_blip import PPO_BLIP as approach

        agent = approach(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True)

    ########################################################################################################################

    tr_reward_arr = []
    te_reward_arr = {}

    for _type in (['mean', 'max', 'min']):
        te_reward_arr[_type] = {}
        for idx in range(len(taskcla)):
            te_reward_arr[_type]['task' + str(idx)] = []

    for task_idx,env_name in task_sequences:
        print(env_name)
        # renew optimizer
        agent.renew_optimizer()

        envs = make_vec_envs(env_name, args.seed, args.num_processes,
                             args.gamma, args.log_dir, device, False)
        obs = envs.reset()

        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                      obs_shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)
        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

        train_ppo(actor_critic, agent, rollouts, task_idx, env_name, task_sequences, envs, obs_shape, args, 
                  episode_rewards, tr_reward_arr, te_reward_arr, num_updates, log_name, device)

        # post-processing
        if args.approach == 'fine-tuning':
            if args.single_task == True:
                envs.close()
                break
            else:
                envs.close()
        elif args.approach == 'ft-fix':
            # fix the backbone
            for param in actor_critic.features.parameters():
                param.requires_grad = False
            if args.single_task == True:
                envs.close()
                break
            else:
                envs.close()
        elif args.approach == 'ewc':
            agent.update_fisher(rollouts, task_idx)
            envs.close()
        elif args.approach == 'blip':
            agent.ng_post_processing(rollouts, task_idx)
            # save the model here so that bit allocation is saved
            save_path = os.path.join(args.save_dir, args.algo)
            torch.save(actor_critic.state_dict(),
                os.path.join(save_path, log_name + '_task_' + str(task_idx) + ".pth"))
            envs.close()

    ########################################################################################################################

if __name__ == '__main__':
    main()
