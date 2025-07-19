import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBufferOld2:
    """
    This ReplayBuffer is for offline RL algorithms only.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu",
        include_mask=False,
        mask_size=0,
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.include_mask = include_mask
        self.mask_size = mask_size

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros(
            (self._max_size,) + self.obs_shape, dtype=obs_dtype
        )
        self.next_observations = np.zeros(
            (self._max_size,) + self.obs_shape, dtype=obs_dtype
        )
        self.next_n_observations = np.zeros(
            (self._max_size,) + self.obs_shape, dtype=obs_dtype
        )
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.next_n_rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.next_n_terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.masks = np.zeros((self._max_size, self.mask_size), dtype=obs_dtype)
        self.next_masks = np.zeros((self._max_size, self.mask_size), dtype=obs_dtype)
        self.next_n_masks = np.zeros((self._max_size, self.mask_size), dtype=obs_dtype)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray,
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy().view(self.action_dim)
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy().reshape(-1, self.action_dim)
        self.rewards[indexes] = np.array(rewards).copy().reshape(-1, 1)
        self.terminals[indexes] = np.array(terminals).copy().reshape(-1, 1)

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def add_batch_with_mask(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        next_n_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_n_rewards: np.ndarray,
        terminals: np.ndarray,
        next_n_terminals: np.ndarray,
        masks: np.ndarray,
        next_masks: np.ndarray,
        next_n_masks: np.ndarray,
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.next_n_observations[indexes] = np.array(next_n_obss).copy()
        self.actions[indexes] = np.array(actions).copy().reshape(-1, self.action_dim)
        self.rewards[indexes] = np.array(rewards).copy().reshape(-1, 1)
        self.next_n_rewards[indexes] = np.array(next_n_rewards).copy().reshape(-1, 1)
        self.terminals[indexes] = np.array(terminals).copy().reshape(-1, 1)
        self.next_n_terminals[indexes] = (
            np.array(next_n_terminals).copy().reshape(-1, 1)
        )
        self.masks[indexes] = np.array(masks).copy().reshape(-1, self.mask_size)
        self.next_masks[indexes] = (
            np.array(next_masks).copy().reshape(-1, self.mask_size)
        )
        self.next_n_masks[indexes] = (
            np.array(next_n_masks).copy().reshape(-1, self.mask_size)
        )
        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32)
        terminals = np.array(dataset["terminals"], dtype=np.float32)
        self.add_batch(observations, next_observations, actions, rewards, terminals)

        # self.observations = observations
        # self.next_observations = next_observations
        # self.actions = actions
        # self.rewards = rewards
        # self.terminals = terminals

        # self._ptr = len(observations)
        # self._size = len(observations)

    def load_dataset_with_mask(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(
            dataset["next_n_observations"], dtype=self.obs_dtype
        )
        next_n_observations = np.array(
            dataset["next_observations"], dtype=self.obs_dtype
        )
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32)
        terminals = np.array(dataset["terminals"], dtype=np.float32)
        next_n_rewards = np.array(dataset["next_n_rewards"], dtype=np.float32)
        next_n_terminals = np.array(dataset["next_n_terminals"], dtype=np.float32)
        masks = np.array(dataset["mask"], dtype=np.float32)
        next_masks = np.array(dataset["next_mask"], dtype=np.float32)
        next_n_masks = np.array(dataset["next_n_mask"], dtype=np.float32)

        self.add_batch_with_mask(
            observations,
            next_observations,
            next_n_observations,
            actions,
            rewards,
            next_n_rewards,
            terminals,
            next_n_terminals,
            masks,
            next_masks,
            next_n_masks,
        )

        # self.observations = observations
        # self.next_observations = next_observations
        # self.actions = actions
        # self.rewards = rewards
        # self.terminals = terminals

        # self._ptr = len(observations)
        # self._size = len(observations)

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(
        self, batch_size: int, with_mask: bool = False
    ) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        if with_mask:
            return {
                "observations": torch.tensor(self.observations[batch_indexes]).to(
                    self.device
                ),
                "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
                "next_observations": torch.tensor(
                    self.next_observations[batch_indexes]
                ).to(self.device),
                "next_n_observations": torch.tensor(
                    self.next_n_observations[batch_indexes]
                ).to(self.device),
                "terminals": torch.tensor(self.terminals[batch_indexes]).to(
                    self.device
                ),
                "next_n_terminals": torch.tensor(
                    self.next_n_terminals[batch_indexes]
                ).to(self.device),
                "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
                "next_n_rewards": torch.tensor(self.next_n_rewards[batch_indexes]).to(
                    self.device
                ),
                "masks": torch.tensor(self.masks[batch_indexes]).to(self.device),
                "next_masks": torch.tensor(self.next_masks[batch_indexes]).to(
                    self.device
                ),
                "next_n_masks": torch.tensor(self.next_n_masks[batch_indexes]).to(
                    self.device
                ),
            }

        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(
                self.device
            ),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(
                self.device
            ),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device),
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[: self._size].copy(),
            "actions": self.actions[: self._size].copy(),
            "next_observations": self.next_observations[: self._size].copy(),
            "terminals": self.terminals[: self._size].copy(),
            "rewards": self.rewards[: self._size].copy(),
        }
