import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from .a2c_ppo_acktr.utils import init

# quant layer specific
from .quant_layer import Conv2d_Q, Linear_Q

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                hidden_size = 1024
            elif len(obs_shape) == 1:
                base = MLPBase
                hidden_size = 64
            else:
                raise NotImplementedError

        self.features = base(obs_shape[0], **base_kwargs)

        # value critic for each task
        self.critic = nn.ModuleList()
        for _ in range(len(taskcla)):
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
            self.critic.append(init_(nn.Linear(hidden_size, 1)))

        # action distribution for each task
        self.dist = nn.ModuleList()
        for taskid, num_outputs in taskcla:
            self.dist.append(Categorical(self.features.output_size, num_outputs))

    @property
    def is_recurrent(self):
        return self.features.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.features.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, taskid, deterministic=False):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, taskid):
        actor_features, _ = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, taskid):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=1024):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 128, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(128 * 7 * 7, hidden_size)), nn.ReLU())

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))

        # self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        # return self.critic_linear(x), x, rnn_hxs
        return x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        # return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
        return hidden_actor, rnn_hxs

class QPolicy(nn.Module):
    def __init__(self, obs_shape, taskcla, base=None, base_kwargs=None):
        super(QPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = QCNNBase
                hidden_size = 1024
            elif len(obs_shape) == 1:
                base = QMLPBase
                hidden_size = 64
            else:
                raise NotImplementedError

        self.features = base(obs_shape[0], **base_kwargs)

        # value critic for each task
        self.critic = nn.ModuleList()
        for _ in range(len(taskcla)):
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0))
            self.critic.append(init_(nn.Linear(hidden_size, 1)))

        # action distribution for each task
        self.dist = nn.ModuleList()
        for taskid, num_outputs in taskcla:
            self.dist.append(Categorical(self.features.output_size, num_outputs))

    @property
    def is_recurrent(self):
        return self.features.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.features.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, taskid, deterministic=False):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, taskid):
        actor_features, _ = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, taskid):
        actor_features, rnn_hxs = self.features(inputs, rnn_hxs, masks)
        value = self.critic[taskid](actor_features)
        dist = self.dist[taskid](actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

class QCNNBase(NNBase):
    def __init__(self, num_inputs, F_prior, recurrent=False, hidden_size=1024):
        super(QCNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), nn.init.calculate_gain('relu'))

        # previous setup, the same as arch reported in EWC
        self.main = nn.Sequential(
            Conv2d_Q(num_inputs, 32, 8, stride=4, F_prior=F_prior), nn.ReLU(),
            Conv2d_Q(32, 64, 4, stride=2, F_prior=F_prior), nn.ReLU(),
            Conv2d_Q(64, 128, 3, stride=1, F_prior=F_prior), nn.ReLU(), Flatten(),
            Linear_Q(128 * 7 * 7, hidden_size, F_prior=F_prior), nn.ReLU())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return x, rnn_hxs


class QMLPBase(NNBase):
    def __init__(self, num_inputs, F_prior, recurrent=False, hidden_size=64):
        super(QMLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            Linear_Q(num_inputs, hidden_size, F_prior=F_prior), nn.Tanh(),
            Linear_Q(hidden_size, hidden_size, F_prior=F_prior), nn.Tanh())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_actor = self.actor(x)

        return hidden_actor, rnn_hxs
