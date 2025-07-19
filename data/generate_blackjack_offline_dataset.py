from environment import BlackJackEnv, BlackJackEnvWrapper
from agent import BlackJackDiscretized2, BlackJackTabular, BlackJackBasic
from model import BlackJackNet5a, BlackJackNet4, BlackJackNet5b
from utils import *
import numpy as np
import torch
import os
import json
from multiprocessing import Pool


def suboptimal_policy1(obs):
    """
    Suboptimal policy for BlackJack
    """
    player_value, usable_ace, dealer_first_card = obs
    if usable_ace == 1:
        if dealer_first_card % 2 == 0:
            return 0 if player_value >= 16 else 1
        else:
            return 0 if player_value >= 20 else 1
    else:
        if dealer_first_card % 2 == 0:
            return 0 if player_value >= 18 else 1
        else:
            return 0 if player_value >= 14 else 1


def suboptimal_policy2(obs):
    """
    Suboptimal policy for BlackJack
    """
    player_value, usable_ace, dealer_first_card = obs
    if usable_ace == 1:
        if dealer_first_card % 2 == 0:
            return 0 if player_value >= 14 else 1
        else:
            return 0 if player_value >= 18 else 1
    else:
        if dealer_first_card % 2 == 0:
            return 0 if player_value >= 19 else 1
        else:
            return 0 if player_value >= 17 else 1


def workload(j, rounds):
    save_path = "./data/BlackJack_dataset/suboptimal_1_offline"  # Save the dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    config = {"agent_type": BlackJackBasic}
    env = BlackJackEnvWrapper(config)
    config_feature = {"agent_type": BlackJackTabular}
    env_feature = BlackJackEnvWrapper(config_feature)
    final_obs_list = []
    final_reward_list = []
    final_done_list = []
    final_mask_list = []
    final_action_list = []
    final_next_obs_list = []
    final_next_action_list = []
    for i in range(rounds):
        obs_list = []
        reward_list = []
        done_list = []
        mask_list = []
        action_list = []
        obs, reward, done = env.reset(init_seed=i + rounds * j)
        obs_feature, reward_feature, done_feature = env_feature.reset(
            init_seed=i + rounds * j
        )
        obs_list.append(obs_feature)
        mask_list.append([1, 1])
        step = 0

        while not done:
            step += 1
            action_mask = [1, 1]

            action = suboptimal_policy1(obs)
            action_list.append(action)
            obs, reward, done = env.step(action)
            obs_feature, reward_feature, done_feature = env_feature.step(action)
            obs_list.append(obs_feature)
            reward_list.append(reward_feature)
            done_list.append(done_feature)
            mask_list.append(action_mask)

        # process the data for offline
        next_obs_list = obs_list[1:]
        next_action_list = action_list[1:] + [action_list[-1]]
        obs_list = obs_list[:-1]
        mask_list = mask_list[:-1]
        # print(
        #     len(next_action_list),
        #     len(action_list),
        #     len(mask_list),
        #     len(obs_list),
        #     len(next_obs_list),
        #     len(reward_list),
        #     len(done_list),
        # )
        final_obs_list.extend(obs_list)
        final_next_obs_list.extend(next_obs_list)
        final_action_list.extend(action_list)
        final_next_action_list.extend(next_action_list)
        final_reward_list.extend(reward_list)
        final_done_list.extend(done_list)
        final_mask_list.extend(mask_list)
        assert (
            len(final_obs_list)
            == len(final_next_obs_list)
            == len(final_next_action_list)
            == len(final_reward_list)
            == len(final_mask_list)
        )
    data = {
        "obs": final_obs_list,
        "action": final_action_list,
        "next_obs": final_next_obs_list,
        "next_action": final_next_action_list,
        "reward": final_reward_list,
        "done": final_done_list,
        "mask": final_mask_list,
    }
    with open(os.path.join(save_path, f"{j}.npz"), "wb") as f:
        np.savez(f, **data)


if __name__ == "__main__":
    pool = Pool(50)
    for i in range(50):
        pool.apply_async(workload, args=(i, 6200))
    # workload(0, 1000)
    pool.close()
    pool.join()
