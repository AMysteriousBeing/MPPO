from environment import BlackJackEnv, BlackJackEnvWrapper
from agent import BlackJackDiscretized2, BlackJackTabular, BlackJackBasic
from model import BlackJackNet5a, BlackJackNet4, BlackJackNet5b
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
    save_path = "./data/BlackJack_dataset/suboptimal_1_all"  # Save the dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    agent = BlackJackBasic()
    config = {"agent_type": BlackJackBasic, "higher_ace_chance": 0.25}
    env = BlackJackEnvWrapper(config)
    data_counter = 0
    for i in range(50000):
        obs, reward, done = env.reset(i)
        action_list = []
        while not done:
            action = suboptimal_policy1(obs)
            action_list.append(action)
            obs, reward, done = env.step(action)
        trajectory = {"seed": i, "action_list": action_list}
        with open(os.path.join(save_path, f"{data_counter}.json"), "w") as f:
            json.dump(trajectory, f)
        data_counter += 1
