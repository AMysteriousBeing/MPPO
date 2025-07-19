from environment import BlackJackEnv, BlackJackEnvWrapper
from agent import BlackJackDiscretized2, BlackJackTabular, BlackJackBasic
from utils import *
import numpy as np
import torch
import os
import json


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


if __name__ == "__main__":
    save_path = "./data/BlackJack_dataset/suboptimal_1_new2"  # Save the dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    agent = BlackJackBasic()
    config = {"agent_type": BlackJackBasic, "higher_ace_chance": 0.25}
    env = BlackJackEnvWrapper(config)
    data_counter = 0
    data_survey = {}
    for ace in [0, 1]:
        data_survey[ace] = {}
        for dealer_first_card in range(1, 11):
            data_survey[ace][dealer_first_card] = {}
            for player_value in range(10, 22):
                data_survey[ace][dealer_first_card][player_value] = 0

    for i in range(1000000):
        next_obs, reward, done = env.reset(i)
        action_list = []
        obs_list = []
        while not done:
            obs = next_obs
            obs_list.append(obs)
            action = suboptimal_policy1(obs)
            action_list.append(action)
            next_obs, reward, done = env.step(action)
        if reward > 0:

            obs = obs_list[0]
            player_value, usable_ace, dealer_first_card = obs
            if player_value < 10:
                substitute_value = 10
            else:
                substitute_value = player_value.item()
            if (
                data_survey[usable_ace.item()][dealer_first_card.item()][
                    substitute_value
                ]
                < 500
            ):
                trajectory = {"seed": i, "action_list": action_list}
                with open(os.path.join(save_path, f"{data_counter}.json"), "w") as f:
                    json.dump(trajectory, f)
                data_counter += 1
                for obs in obs_list:
                    player_value, usable_ace, dealer_first_card = obs
                    if player_value < 10:
                        substitute_value = 10
                    else:
                        substitute_value = player_value.item()
                    data_survey[usable_ace.item()][dealer_first_card.item()][
                        substitute_value
                    ] += 1
    with open(os.path.join("./", f"survey10x_subopt1_new.json"), "w") as f:
        json.dump(data_survey, f, indent=2)
