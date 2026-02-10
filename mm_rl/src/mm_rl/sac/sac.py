"""Soft Actor-Critic (SAC) algorithm implementation."""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mm_rl.sac.networks import Actor, DoubleCritic


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in batch]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class SAC:
    """Soft Actor-Critic algorithm."""

    def __init__(
        self,
        state_dim,
        action_dim,
        action_range=(-1.0, 1.0),
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        hidden_dim=256,
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)

        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy coefficient
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

    @property
    def alpha_value(self):
        if self.auto_alpha:
            return self.log_alpha.exp().item()
        return self.alpha

    def select_action(self, state, deterministic=False):
        """Select action from policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor(state)
            action = torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(state)
        action = action.detach().numpy()[0]
        # Scale action to action range
        action = self._scale_action(action)
        return action

    def _scale_action(self, action):
        """Scale action from [-1, 1] to [action_range[0], action_range[1]]."""
        low, high = self.action_range
        return low + (action + 1.0) * 0.5 * (high - low)

    def _unscale_action(self, action):
        """Unscale action from [action_range[0], action_range[1]] to [-1, 1]."""
        low, high = self.action_range
        return 2.0 * (action - low) / (high - low) - 1.0

    def update(self, batch_size=256):
        """Update networks using a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return {}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Unscale actions for network (networks expect [-1, 1])
        actions_unscaled = self._unscale_action(actions)

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # Compute target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = (
                torch.min(target_q1, target_q2) - self.alpha_value * next_log_probs
            )
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions_unscaled)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(
            current_q2, target_q
        )

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha_value * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha if auto-tuning
        alpha_loss = None
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update target network
        self._soft_update(self.critic_target, self.critic, self.tau)

        # Return metrics
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": q_new.mean().item(),
            "alpha": self.alpha_value,
        }
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()

        return metrics

    def _soft_update(self, target, source, tau):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, filepath):
        """Save model checkpoints."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha if self.auto_alpha else None,
                "alpha_optimizer": (
                    self.alpha_optimizer.state_dict() if self.auto_alpha else None
                ),
            },
            filepath,
        )

    def load(self, filepath):
        """Load model checkpoints."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.auto_alpha and checkpoint["log_alpha"] is not None:
            self.log_alpha.data = checkpoint["log_alpha"].data
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
