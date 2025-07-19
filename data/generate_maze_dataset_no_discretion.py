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


def workload(i_start, i_end):
    save_path = "./data/Maze_dataset/right_wall_policy_all"  # Save the dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    env = MazeEnv2({"agent_type": MazeAgentPassThrough, "debug": False})
    data_counter = i_start
    for i in range(i_start, i_end):
        obs, reward, done = env.reset(init_seed=i)
        step = 0
        action_list = []
        while not done:
            step += 1
            map_obs = obs["cnn_obs"]
            pass_through_obs = obs["dense_obs"]
            action_mask = obs["action_mask"]
            action = right_wall_policy(map_obs, pass_through_obs, action_mask)
            action_list.append(action)
            obs, reward, done = env.step(action)
        trajectory = {"seed": i, "action_list": action_list}
        with open(os.path.join(save_path, f"{data_counter}.json"), "w") as f:
            json.dump(trajectory, f)
        data_counter += 1


if __name__ == "__main__":
    pool = Pool(50)
    for i in range(50):
        pool.apply_async(workload, args=(1000 * i, 1000 * (i + 1)))
    pool.close()
    pool.join()
