import numpy as np
import torch
import collections
import os

# qlearning_dataset with n_step info


def qlearning_dataset2(
    path_to_data,
    world_size,
    rank,
    include_mask=False,
    create_artificial_next_mask=False,
    **kwargs
):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    obs_ = []
    next_obs_ = []
    next_n_obs_ = []
    action_ = []
    reward_ = []
    next_n_reward_ = []
    done_ = []
    next_n_done_ = []
    mask_ = []
    next_mask_ = []
    next_n_mask_ = []
    file_list = os.listdir(path_to_data)
    file_list.sort()
    len_list = len(file_list)
    segment_length = len_list // world_size

    for file in file_list[rank * segment_length : (rank + 1) * segment_length]:
        # for file in file_list[:1]:
        path_to_file = os.path.join(path_to_data, file)
        npz = np.load(path_to_file)
        obs_.extend(npz["obs"])
        next_obs_.extend(npz["next_obs"])
        next_n_obs_.extend(npz["next_n_obs"])
        action_.extend(npz["action"])
        reward_.extend(npz["reward"])
        next_n_reward_.extend(npz["next_n_reward"])
        done_.extend(npz["done"])
        next_n_done_.extend(npz["next_n_done"])
        if include_mask:
            mask_.extend(npz["mask"])
            if create_artificial_next_mask:
                next_mask_.extend(np.ones_like(npz["mask"]))
                next_n_mask_.extend(np.ones_like(npz["mask"]))
            else:
                next_mask_.extend(npz["next_mask"])
                next_n_mask_.extend(npz["next_n_mask"])

    if include_mask:
        return {
            "observations": np.array(obs_),
            "actions": np.array(action_),
            "next_observations": np.array(next_obs_),
            "next_n_observations": np.array(next_n_obs_),
            "rewards": np.array(reward_),
            "next_n_rewards": np.array(next_n_reward_),
            "terminals": np.array(done_),
            "next_n_terminals": np.array(next_n_done_),
            "mask": np.array(mask_),
            "next_mask": np.array(next_mask_),
            "next_n_mask": np.array(next_n_mask_),
        }

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


if __name__ == "__main__":
    path_to_data = "../data/smth_offline_dataset_consolidated"
    dataset = qlearning_dataset(path_to_data)
    print(dataset["observations"].shape)
