import os
import time
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch

from .a2c_ppo_acktr import utils
from .evaluation import evaluate

def train_ppo(actor_critic, agent, rollouts, task_idx, env_name, task_sequences, envs, obs_shape, args,
              episode_rewards, tr_reward_arr, te_reward_arr, num_updates, log_name, device):
    # evaluate here so that we can conveniently plot
    ob_rms = None
    eval_episode_mean_rewards = evaluate(actor_critic, ob_rms, task_sequences, args.seed,
                            10, args.log_dir, device, obs_shape, task_idx, args.gamma)
    print ('len task_sequences : ', len(task_sequences))
    for idx in range(len(task_sequences)):
        te_reward_arr['mean']['task' + str(idx)].append((eval_episode_mean_rewards[idx]))
    sio.savemat('./result_data/'+log_name + '_result.mat',{'tr_reward_arr':np.array(tr_reward_arr),
                                                                     'te_reward_arr':np.array(te_reward_arr)})

    start = time.time()
    for j in tqdm(range(num_updates)):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], task_idx)

            # print (action.shape)
            # envs.render() # render the environment
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            # if args.experiment == 'roboschool':
            #     new_obs[:, :obs_shape_real[0]] = obs
            new_obs = obs
            rollouts.insert(new_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], task_idx).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, task_idx)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if ((j+1) % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save(actor_critic.state_dict(),
                os.path.join(save_path, log_name + '_task_' + str(task_idx) + ".pth"))

        if (j+1) % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

            tr_reward_arr.append(np.mean(episode_rewards))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                    and (j+1) % args.eval_interval == 0):
            ob_rms = None
            eval_episode_mean_rewards = evaluate(actor_critic, ob_rms, task_sequences, args.seed,
                            10, args.log_dir, device, obs_shape, task_idx, args.gamma)

            print ('len task_sequences : ', len(task_sequences))
            for idx in range(len(task_sequences)):
                te_reward_arr['mean']['task' + str(idx)].append((eval_episode_mean_rewards[idx]))
            sio.savemat('./result_data/'+log_name + '_result.mat',{'tr_reward_arr':np.array(tr_reward_arr),
                                                                     'te_reward_arr':np.array(te_reward_arr)})
