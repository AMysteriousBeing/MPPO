import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import copy
from typing import Dict, Union, Tuple
from copy import deepcopy
from torch.distributions import Categorical


class CQLDiscretePolicy(nn.Module):
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        action_size: int,
        learning_rate: float,
        device: str,
        tau: float = 0.002,
        gamma: float = 0.99,
        cql_weight: float = 1.0,
        with_lagrange: bool = True,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super(CQLDiscretePolicy, self).__init__()
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        learning_rate = learning_rate
        self.clip_grad_param = 1
        self.device = device
        self.target_action_gap = 0.0

        self.target_entropy = -action_size  # -dim(A)

        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = with_lagrange
        self.cql_weight = cql_weight
        self.cql_log_alpha = torch.zeros(1, requires_grad=True)
        self.cql_alpha_optimizer = optim.Adam(
            params=[self.cql_log_alpha], lr=learning_rate
        )

        # Actor Network

        self.actor_local = actor
        self.actor_optimizer = actor_optim

        # Critic Network (w/ Target Network)

        self.critic1 = critic1
        self.critic2 = critic2

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = deepcopy(critic1)

        self.critic2_target = deepcopy(critic2)

        self.critic1_optimizer = critic1_optim
        self.critic2_optimizer = critic2_optim
        self.softmax = nn.Softmax(dim=-1)

    def get_action(self, state, eval=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        with torch.no_grad():
            action = self.actor_local.get_det_action(state)
        return action.numpy()

    def train(self) -> None:
        self.actor_local.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor_local.eval()
        self.critic1.eval()
        self.critic2.eval()

    def calc_policy_loss(self, states, alpha):
        action_probs = self.actor_local(states)

        dist = Categorical(action_probs)
        action = dist.sample()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_pis = torch.log(action_probs + z)

        q1 = self.critic1(states)
        q2 = self.critic2(states)
        min_Q = torch.min(q1, q2)
        actor_loss = (
            (action_probs * (alpha.to(self.device) * log_pis - min_Q)).sum(1).mean()
        )
        log_action_pi = torch.sum(log_pis * action_probs, dim=1)
        return actor_loss, log_action_pi

    def learn(self, batch):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, next_states, rewards, dones = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            batch["terminals"],
        )
        actions = torch.nn.functional.one_hot(actions, num_classes=235)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)
        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = -(
            self.log_alpha.exp() * (log_pis.cpu() + self.target_entropy).detach().cpu()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            action_probs = self.actor_local(states)

            dist = Categorical(action_probs)
            action = dist.sample()
            # Have to deal with situation of 0.0 probabilities because we can't do log 0
            z = action_probs == 0.0
            z = z.float() * 1e-8
            log_pis = torch.log(action_probs + z)
            Q_target1_next = self.critic1_target(next_states)
            Q_target2_next = self.critic2_target(next_states)
            Q_target_next = action_probs * (
                torch.min(Q_target1_next, Q_target2_next)
                - self.alpha.to(self.device) * log_pis
            )

            # Compute Q targets for current states (y_i)

            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next)
            # print(
            #     Q_targets.shape,
            #     rewards.shape,
            #     dones.shape,
            #     Q_target_next.shape,
            #     Q_target_next.sum(dim=1).unsqueeze(-1).shape,
            # )

        # Compute critic loss
        q1 = self.critic1(states)
        q2 = self.critic2(states)

        q1_ = q1.gather(1, actions.long())
        q2_ = q2.gather(1, actions.long())

        critic1_loss = 0.5 * F.mse_loss(q1_, Q_targets)
        critic2_loss = 0.5 * F.mse_loss(q2_, Q_targets)

        cql1_scaled_loss = torch.logsumexp(q1, dim=1).mean() - q1.mean()
        cql2_scaled_loss = torch.logsumexp(q2, dim=1).mean() - q2.mean()

        cql_alpha_loss = torch.FloatTensor([0.0])
        cql_alpha = torch.FloatTensor([0.0])
        if self.with_lagrange:
            cql_alpha = torch.clamp(
                self.cql_log_alpha.exp(), min=0.0, max=1000000.0
            ).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (-cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "loss/cql_alpha": cql_alpha_loss.item(),
            "cql_alpha": cql_alpha.item(),
        }

        return result

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
