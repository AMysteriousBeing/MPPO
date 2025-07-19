from multiprocessing import Process, shared_memory
from utils import load_actions, data_augmentation
import torch
import uuid
import time
import json
import os
import numpy as np
import random
from agent import ZombieAgent, SearchAgent, SearchAgentSlim, FeatureAgent2Adapted
from environment import MahjongGBEnv
from multiprocessing import Pool

LOG_FORMAT = (
    "%(levelname) -8s %(asctime)s %(name) -25s %(funcName) "
    "-25s %(lineno) -5d: %(message)s"
)
TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]


def generate_data(
    file_name_list,
    config,
):

    # need write to config
    data_dir = config["path_to_data"]
    output_dir = config["output_dir"]
    torch.set_num_threads(1)
    game_counter = 0
    for file_name in file_name_list:
        # load game data from file
        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "r") as f:
            recorded_data = json.load(f)

        # select augmentation key given sampling setup
        if recorded_data["augmentable"] == True:
            augmentation_key = random.randint(0, 11)
        else:
            augmentation_key = 0

        for i in range(4):
            config["agent_clz{}".format(i)] = FeatureAgent2Adapted
        env = MahjongGBEnv(config)
        episode_data = {
            agent_name: {
                "state": {"observation": [], "action_mask": []},
                "action": [],
                "reward": [],
                "value": [],
                "info": [],
            }
            for agent_name in env.agent_names
        }

        # get maximum cut-off step
        cut_off_step = -1
        obs = env.reset(
            recorded_data["wind"],
            data_augmentation(recorded_data["initwall"], augmentation_key),
            cut_off_step,
        )
        done = False
        # load history actions from augmented game setup
        action_history = load_actions(recorded_data["history"], augmentation_key)
        action_pointer = [0, 0, 0, 0]

        step_counter = 6
        game_length = 0
        game_counter += 1
        # random flag for correct action
        correct_action_flag = True
        while not done:
            step_counter += 1
            # each player take action
            actions = {}
            player_name_list = list(obs.keys())
            player_id_list = [int(a[-1]) - 1 for a in player_name_list]
            for i in range(len(player_id_list)):
                player_id = player_id_list[i]
                # infer with network
                agent_name = player_name_list[i]
                agent_data = episode_data[agent_name]
                state = obs[agent_name]
                agent_data["state"]["observation"].append(state["observation"])
                agent_data["state"]["action_mask"].append(state["action_mask"])
                state["observation"] = torch.tensor(
                    state["observation"], dtype=torch.float32
                ).unsqueeze(0)
                state["action_mask"] = torch.tensor(
                    state["action_mask"], dtype=torch.float32
                ).unsqueeze(0)

                # use correct action
                if correct_action_flag:
                    # calculate correct action
                    player_id_action_counter = action_pointer[player_id]
                    action_pointer[player_id] += 1
                    if player_id_action_counter < len(action_history[player_id]):
                        correct_action = action_history[player_id][
                            player_id_action_counter
                        ]
                        action = correct_action
                actions[agent_name] = action
                agent_data["action"].append(actions[agent_name])

            # interact with env
            next_obs, rewards, done = env.step(actions)
            for agent_name in rewards:
                episode_data[agent_name]["reward"].append(rewards[agent_name])
            obs = next_obs
            game_length += 1

        # postprocessing episode data for each agent
        for agent_name, agent_data in episode_data.items():
            if len(agent_data["action"]) < len(agent_data["reward"]):
                agent_data["reward"].pop(0)
            if len(agent_data["state"]["observation"]) > 0:
                obs = np.stack(agent_data["state"]["observation"])
                mask = np.stack(agent_data["state"]["action_mask"])
                actions = np.array(agent_data["action"], dtype=np.int64)
                rewards = np.array(agent_data["reward"], dtype=np.float32)
                next_obs = np.stack(
                    agent_data["state"]["observation"][1:],
                    dtype=np.float32,
                )
                next_obs = np.append(
                    next_obs, agent_data["state"]["observation"][0:1], axis=0
                )
                next_mask = np.stack(
                    agent_data["state"]["action_mask"][1:],
                    dtype=np.float32,
                )
                next_mask = np.append(
                    next_mask, agent_data["state"]["action_mask"][0:1], axis=0
                )
                next_actions = np.stack(
                    agent_data["action"][1:],
                    dtype=np.int64,
                )
                next_actions = np.append(
                    next_actions, agent_data["action"][0:1], axis=0
                )
                is_done = [0] * len(rewards)
                is_done[-1] = 1
                is_done = np.array(is_done, dtype=np.int64)

                # final_reward = []
                # for i in range(len(rewards)):
                #     final_reward.append(sum(rewards[i:]))
                # final_reward = np.array(final_reward, dtype=np.float32)
                # filter data
                final_obs = []
                final_action = []
                final_next_obs = []
                final_next_actions = []
                final_rewards = []
                final_is_done = []
                final_mask = []
                final_next_mask = []
                for i in range(len(rewards)):
                    if sum(mask[i]) > 1:
                        final_obs.append(obs[i])
                        final_action.append(actions[i])
                        final_next_obs.append(next_obs[i])
                        final_next_actions.append(next_actions[i])
                        final_rewards.append(rewards[i])
                        final_is_done.append(is_done[i])
                        final_mask.append(mask[i])
                        final_next_mask.append(next_mask[i])
                # convert to numpy
                final_obs = np.array(final_obs, dtype=np.float32)
                final_action = np.array(final_action, dtype=np.int64)
                final_next_obs = np.array(final_next_obs, dtype=np.float32)
                final_next_actions = np.array(final_next_actions, dtype=np.int64)
                final_rewards = np.array(final_rewards, dtype=np.float32)
                final_is_done = np.array(final_is_done, dtype=np.int64)
                final_mask = np.array(final_mask, dtype=np.float32)
                final_next_mask = np.array(final_next_mask, dtype=np.float32)
                # send samples to replay_buffer (per agent)
                data = {
                    "obs": final_obs,
                    "action": final_action,
                    "next_obs": final_next_obs,
                    "next_actions": final_next_actions,
                    "rewards": final_rewards,
                    "is_done": final_is_done,
                    "action_mask": final_mask,
                    "next_action_mask": final_next_mask,
                }
                with open(
                    os.path.join(output_dir, f"{file_name}_{agent_name}.npz"),
                    "wb",
                ) as f:
                    np.savez(f, **data)


if __name__ == "__main__":

    config = {
        "path_to_data": "data/smth/smth_pass1",
        "init_hands": 31400,
        "augmentation": False,
        "output_dir": "data/smth_offline_dataset",
    }
    if not os.path.exists(config["output_dir"]):
        os.makedirs(config["output_dir"])
    file_name_list_total = os.listdir(config["path_to_data"])
    list_of_list = []
    cpu_count = 50
    len_segment = 15000 // cpu_count
    for i in range(cpu_count):
        list_of_list.append(
            file_name_list_total[i * len_segment : (i + 1) * len_segment]
        )
    pool = Pool(cpu_count)
    for file_name_list in list_of_list:
        pool.apply_async(generate_data, args=(file_name_list, config))
    pool.close()
    pool.join()
    # generate_data(["0.json"], config)
