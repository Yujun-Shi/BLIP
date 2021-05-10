import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO_EWC():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 ewc_epoch = 1,
                 ewc_lambda = 5000,
                 online = False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.ewc_epoch = 1
        self.ewc_lambda = ewc_lambda
        
        print ('ewc_lambda : ', self.ewc_lambda)

        # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.lr = lr
        self.eps = eps
        self.EWC_task_count = 0
        self.divide_factor = 0
        self.online = online

    def renew_optimizer(self):
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr, eps=self.eps)

    def update(self, rollouts, task_num):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch, task_num)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                reg_loss = self.ewc_lambda * self.ewc_loss()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef + reg_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def estimate_fisher(self, rollouts, task_num):
        fisher_dict = {}
        for n,p in self.actor_critic.features.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                fisher_dict[n] = torch.zeros_like(p)
        est_fisher_info = fisher_dict

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        total_batch = 0
        for _ in range(self.ewc_epoch):
            # set number of mini-batch to 32
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, 32)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, 32)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, _, \
                   _, _, masks_batch, _, _ = sample

                batch_size_t = obs_batch.shape[0]
                total_batch += batch_size_t

                # clear gradient
                self.actor_critic.zero_grad()

                # get action distribution
                actor_features, _ = self.actor_critic.features(obs_batch, 
                    recurrent_hidden_states_batch, masks_batch)
                batch_action_dist = self.actor_critic.dist[task_num](actor_features)
                sampled_actions = batch_action_dist.sample()
                sampled_action_log_probs = batch_action_dist.log_probs(sampled_actions)
                (-sampled_action_log_probs.mean()).backward()

                for n, p in self.actor_critic.features.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher_info[n] += batch_size_t*(p.grad.detach() ** 2)

        for n, p in self.actor_critic.features.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] /= float(total_batch)
        return est_fisher_info

    # def estimate_fisher(self, rollouts, task_num):
    #     fisher_dict = {}
    #     for n,p in self.actor_critic.named_parameters():
    #         if p.requires_grad:
    #             n = n.replace('.', '__')
    #             fisher_dict[n] = torch.zeros_like(p)
    #     est_fisher_info = fisher_dict

    #     advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    #     advantages = (advantages - advantages.mean()) / (
    #         advantages.std() + 1e-5)

    #     for e in range(self.ewc_epoch):
    #         if self.actor_critic.is_recurrent:
    #             data_generator = rollouts.recurrent_generator(
    #                 advantages, self.num_mini_batch)
    #         else:
    #             data_generator = rollouts.feed_forward_generator(
    #                 advantages, self.num_mini_batch)

    #         for sample in data_generator:
    #             obs_batch, recurrent_hidden_states_batch, actions_batch, \
    #                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
    #                     adv_targ = sample

    #             # Reshape to do in a single forward pass for all steps
    #             values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
    #                 obs_batch, recurrent_hidden_states_batch, masks_batch,
    #                 actions_batch, task_num)

    #             ratio = torch.exp(action_log_probs -
    #                               old_action_log_probs_batch)
    #             surr1 = ratio * adv_targ
    #             surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
    #                                 1.0 + self.clip_param) * adv_targ
    #             action_loss = -torch.min(surr1, surr2).mean()

    #             if self.use_clipped_value_loss:
    #                 value_pred_clipped = value_preds_batch + \
    #                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
    #                 value_losses = (values - return_batch).pow(2)
    #                 value_losses_clipped = (
    #                     value_pred_clipped - return_batch).pow(2)
    #                 value_loss = 0.5 * torch.max(value_losses,
    #                                              value_losses_clipped).mean()
    #             else:
    #                 value_loss = 0.5 * (return_batch - values).pow(2).mean()

    #             self.optimizer.zero_grad()
    #             (value_loss * self.value_loss_coef + action_loss -
    #              dist_entropy * self.entropy_coef).backward()
    #             nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
    #                                      self.max_grad_norm)

    #             for n, p in self.actor_critic.named_parameters():
    #                 if p.requires_grad:
    #                     n = n.replace('.', '__')
    #                     if p.grad is not None:
    #                         est_fisher_info[n] += p.grad.detach() ** 2

    #     return est_fisher_info

    def store_fisher_n_params(self, fisher):
        # Store new values in the network
        for n, p in self.actor_critic.features.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                # -mode (=MAP parameter estimate)
                self.actor_critic.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self.actor_critic, '{}_EWC_estimated_fisher'.format(n))
                    # yujun: simply divided by number of tasks instead of using moving average
                    # fisher[n] += self.gamma * existing_values
                    fisher[n] += existing_values
                self.actor_critic.register_buffer('{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     fisher[n])

        # If "offline EWC", increase task-count (for "online EWC", set it to 1 to indicate EWC-loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1
        # self.divide_factor += 1

    def update_fisher(self, rollouts, task_num):
        fisher = self.estimate_fisher(rollouts, task_num)
        self.store_fisher_n_params(fisher)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.actor_critic.features.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self.actor_critic, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self.actor_critic, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        # yujun: simply divided by number of tasks
                        # fisher = self.gamma * fisher if self.online else fisher
                        # fisher = fisher / self.divide_factor
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return 0.
