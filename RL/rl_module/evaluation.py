import numpy as np
import torch
from collections import deque
from .a2c_ppo_acktr import utils
from .a2c_ppo_acktr.envs import make_vec_envs

def evaluate(actor_critic, ob_rms, task_sequences, seed, num_processes, eval_log_dir,
             device, obs_shape, current_task_idx, gamma):

    eval_episode_rewards_arr = []

    for task_idx,task_name in task_sequences:

        if task_idx <= current_task_idx:

            eval_envs = make_vec_envs(task_name, seed + num_processes, num_processes,
                                      gamma, eval_log_dir, device, False)

            eval_episode_rewards = deque(maxlen=10)

            obs = eval_envs.reset()
            obs_shape_real = eval_envs.observation_space.shape
            current_obs = torch.zeros(num_processes, *obs_shape).cuda()
            #### reshape for training ##############
            current_obs[:, :obs_shape_real[0]] = obs
            ########################################
            eval_recurrent_hidden_states = torch.zeros(
                num_processes, actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        current_obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        task_idx,
                        deterministic=True)

                # Obser reward and next obs
                # eval_envs.render()
                obs, _, done, infos = eval_envs.step(action)

                #### reshape for training ###############
                current_obs[:, :obs_shape_real[0]] = obs
                ########################################

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=device)

                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_episode_rewards_arr.append(np.mean(eval_episode_rewards))

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f} \n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards)))
        else:
            eval_episode_rewards_arr.append(0)

    return eval_episode_rewards_arr
