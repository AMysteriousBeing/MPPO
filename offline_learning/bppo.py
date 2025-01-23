import torch
import numpy as np
from buffer import ReplayBuffer
from torch.distributions import Distribution
from copy import deepcopy
import torch.nn as nn

CONST_EPS = 1e-10


def log_prob_func(dist: Distribution, action: torch.Tensor) -> torch.Tensor:
    log_prob = dist.log_prob(action)
    if len(log_prob.shape) == 1:
        return log_prob
    else:
        return log_prob.sum(-1, keepdim=True)


class BPPO(nn.Module):

    def __init__(
        self,
        actor: nn.Module,
        criticq: nn.Module,
        criticv: nn.Module,
        actor_optim: torch.optim.Optimizer,
        learning_rate: float,
        clip_ratio: float,
        entropy_weight: float,
        omega: float,
        batch_size: int,
        device: str,
        tau: float = 0.002,
        gamma: float = 0.99,
    ) -> None:
        super(BPPO, self).__init__()
        self._device = device
        self._policy = actor
        self._optimizer = actor_optim
        self._policy_lr = learning_rate
        self._old_policy = deepcopy(self._policy)
        self._scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=2, gamma=0.98
        )
        self.criticv = criticv
        self.criticq = criticq

        self._clip_ratio = clip_ratio
        self._entropy_weight = entropy_weight
        self._omega = omega
        self._batch_size = batch_size

    def loss(
        self,
        states,
        masks,
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------
        s = states
        old_dist = self._old_policy(s)
        inf_mask = torch.clamp(torch.log(masks), -1e20, 1e20)
        masked_logits = old_dist + inf_mask
        old_dist = torch.distributions.Categorical(logits=masked_logits)
        a = old_dist.sample()
        a_one_hot = torch.nn.functional.one_hot(a, num_classes=235)
        advantage = self.criticq(s, a_one_hot) - self.criticv(s)
        advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(s)
        masked_logits_new = new_dist + inf_mask
        new_dist = torch.distributions.Categorical(logits=masked_logits_new)

        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()

        advantage = self.weighted_advantage(advantage)

        loss1 = ratio * advantage

        loss2 = (
            torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage
        )

        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight

        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss

    # def update(
    #     self,
    #     replay_buffer,
    #     Q: QLearner,
    #     value: ValueLearner,
    #     is_clip_decay: bool,
    # ) -> float:
    #     policy_loss = self.loss(replay_buffer, Q, value, is_clip_decay)

    #     self._optimizer.zero_grad()
    #     policy_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
    #     self._optimizer.step()

    #     return policy_loss.item()

    def learn(self, batch):
        states, actions, next_states, rewards, dones, masks = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["rewards"],
            batch["terminals"],
            batch["masks"],
        )

        policy_loss = self.loss(states, masks)

        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()
        result = {
            "loss/policy": policy_loss.item(),
        }
        return result

    def select_action(self, s: torch.Tensor, is_sample: bool) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:
            action = dist.mean
        # clip
        action = action.clamp(-1.0, 1.0)
        return action

    def set_old_policy(
        self,
    ) -> None:
        self._old_policy.load_state_dict(self._policy.state_dict())

    def weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:
            weight = torch.zeros_like(advantage)
            index = torch.where(advantage > 0)[0]
            weight[index] = self._omega
            weight[torch.where(weight == 0)[0]] = 1 - self._omega
            weight.to(self._device)
            return weight * advantage
