import numpy as np
import torch
from environment import MazeEnv2
from agent import MazeAgentBasic, MazeAgentPassThrough
from environment import MegaMazeGenerator, MazeGenerator2, MazeEnv2
from model import MazeNet2
from utils import *
import os
import json
from multiprocessing import Pool
import copy


def left_wall_policy(obs, dense_obs, action_mask):
    # if there is only one option
    if sum(action_mask) == 1:
        return np.argmax(action_mask).item()
    # decode previous location:
    cur_location = (int(dense_obs[0] * 19 + 0.1), int(dense_obs[1] * 19 + 0.1))
    prev_location = (int(dense_obs[2] * 19 + 0.1), int(dense_obs[3] * 19 + 0.1))
    prev_x, prev_y = (
        prev_location[0] - cur_location[0],
        prev_location[1] - cur_location[1],
    )
    if prev_y < 0:
        action_order = [0, 3, 1, 2]
    elif prev_x < 0:
        action_order = [3, 1, 2, 0]
    elif prev_y > 0:
        action_order = [1, 2, 0, 3]
    elif prev_x > 0:
        action_order = [2, 0, 3, 1]
    for i in action_order:
        if action_mask[i] > 0:
            return i


def right_wall_policy(obs, dense_obs, action_mask):
    # if there is only one option
    if sum(action_mask) == 1:
        return np.argmax(action_mask).item()
    # decode previous location:
    cur_location = (int(dense_obs[0] * 19 + 0.1), int(dense_obs[1] * 19 + 0.1))
    prev_location = (int(dense_obs[2] * 19 + 0.1), int(dense_obs[3] * 19 + 0.1))
    prev_x, prev_y = (
        prev_location[0] - cur_location[0],
        prev_location[1] - cur_location[1],
    )
    if prev_y < 0:
        action_order = [1, 3, 0, 2]
    elif prev_x < 0:
        action_order = [2, 1, 3, 0]
    elif prev_y > 0:
        action_order = [0, 2, 1, 3]
    elif prev_x > 0:
        action_order = [3, 0, 2, 1]
    for i in action_order:
        if action_mask[i] > 0:
            return i


def visualize(data):
    maze_vis = data
    # maze_vis[self.location[0], self.location[1]] = 1
    for row in maze_vis:
        for cell in row:
            if cell == -1:
                print("█", end=" ")
            elif 0 < cell:
                print(int(cell), end=" ")
            elif cell == -2:
                print("~", end=" ")
            else:
                print(" ", end=" ")
        print()


def workload(j, rounds):
    save_path = (
        "./data/Maze_dataset/right_wall_policy_offline_n_step"  # Save the dataset
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    env = MazeEnv2({"agent_type": MazeAgentPassThrough, "debug": False})
    env_feature = MazeEnv2({"agent_type": MazeAgentBasic, "debug": False})
    final_obs_list = []
    final_reward_list = []
    final_next_n_reward_list = []
    final_done_list = []
    final_next_n_done_list = []
    final_mask_list = []
    final_next_mask_list = []
    final_next_n_mask_list = []
    final_action_list = []
    final_next_obs_list = []
    final_next_n_obs_list = []
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
        flattened_feature = torch.concat(
            [obs_feature["cnn_obs"].view(-1), obs_feature["dense_obs"]]
        )
        obs_list.append(flattened_feature.tolist())
        mask_list.append(obs["action_mask"].tolist())
        step = 0

        while not done:
            step += 1
            map_obs = obs["cnn_obs"]
            pass_through_obs = obs["dense_obs"]
            action_mask = obs["action_mask"]

            action = right_wall_policy(map_obs, pass_through_obs, action_mask)
            action_list.append(action)
            obs, reward, done = env.step(action)
            obs_feature, reward_feature, done_feature = env_feature.step(action)
            flattened_feature = torch.concat(
                [obs_feature["cnn_obs"].view(-1), obs_feature["dense_obs"]]
            )
            obs_list.append(flattened_feature.tolist())
            reward_list.append(reward_feature)
            done_list.append(done_feature)
            mask_list.append(obs["action_mask"].tolist())

        # process the data for offline
        next_obs_list = obs_list[1:]
        next_reward_list = []
        for i in range(1, 10):
            next_reward = np.array(reward_list[i:] + [0] * i, dtype=np.float32)
            next_reward_list.append(next_reward)
        td_10_reward = np.array(reward_list)
        for i in range(1, 10):
            td_10_reward += np.array(next_reward_list[i - 1])
        td_10_reward = td_10_reward.tolist()
        next_action_list = action_list[1:] + [action_list[-1]]
        obs_list = obs_list[:-1]
        next_n_obs_list = obs_list[10:] + [obs_list[-1]] * 10
        mask_list = mask_list[:-1]
        next_mask_list = mask_list[1:] + [mask_list[-1]]
        next_n_mask_list = mask_list[10:] + [mask_list[0]] * 10
        next_n_done_list = copy.deepcopy(done_list)
        for i in range(1, 11):
            next_n_done_list[-i] = 1
        final_obs_list.extend(obs_list)
        final_next_obs_list.extend(next_obs_list)
        final_next_n_obs_list.extend(next_n_obs_list)
        final_action_list.extend(action_list)
        final_next_action_list.extend(next_action_list)
        final_reward_list.extend(reward_list)
        final_next_n_reward_list.extend(td_10_reward)
        final_done_list.extend(done_list)
        final_next_n_done_list.extend(next_n_done_list)
        final_mask_list.extend(mask_list)
        final_next_mask_list.extend(next_mask_list)
        final_next_n_mask_list.extend(next_n_mask_list)
        # print(len(final_next_mask_list), len(final_next_n_mask_list))
    data = {
        "obs": final_obs_list,
        "action": final_action_list,
        "next_obs": final_next_obs_list,
        "next_action": final_next_action_list,
        "next_n_obs": final_next_n_obs_list,
        "reward": final_reward_list,
        "next_n_reward": final_next_n_reward_list,
        "done": final_done_list,
        "next_n_done": final_next_n_done_list,
        "mask": final_mask_list,
        "next_mask": final_next_mask_list,
        "next_n_mask": final_next_n_mask_list,
    }

    with open(os.path.join(save_path, f"{j}.npz"), "wb") as f:
        np.savez(f, **data)


if __name__ == "__main__":
    pool = Pool(50)
    for i in range(50):
        pool.apply_async(workload, args=(i, 1000))
    # workload(0, 1000)
    pool.close()
    pool.join()
    # workload(0, 1000)
