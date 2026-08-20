"""Neural network definitions for SAC."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor network for SAC that outputs mean and log_std for actions."""

    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()
        layers = list(hidden_layers)
        dims = [state_dim] + layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.fc_mean = nn.Linear(dims[-1], action_dim)
        self.fc_logstd = nn.Linear(dims[-1], action_dim)

    def forward(self, state):
        x = state
        for fc in self.fcs:
            x = F.relu(fc(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing action bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean


class Critic(nn.Module):
    """Critic network (Q-function) for SAC."""

    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()
        layers = list(hidden_layers)
        dims = [state_dim + action_dim] + layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.fc_out = nn.Linear(dims[-1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        q = self.fc_out(x)
        return q


class DoubleCritic(nn.Module):
    """Double Q-network to reduce overestimation bias."""

    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_layers)
        self.q2 = Critic(state_dim, action_dim, hidden_layers)

    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def q1_forward(self, state, action):
        return self.q1(state, action)
