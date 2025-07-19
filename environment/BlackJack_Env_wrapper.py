import random
from agent import BlackJackAgentTemplate
import gymnasium as gym
import numpy as np


class BlackJackEnvWrapper:
    """
    Dual player Black Jack Environment, as described in Sutton Book Example 5.1
    """

    agent_names = ["player_%d" % i for i in range(1)]

    def __init__(self, config):
        self.env = gym.make("Blackjack-v1", natural=False, sab=True)
        assert "agent_type" in config, "must specify agent_type to process features!"
        self.agent_type = config["agent_type"]
        assert issubclass(
            self.agent_type, BlackJackAgentTemplate
        ), "ageng_type must be a subclass of BlackJackAgentTemplate!"
        self.agent = self.agent_type()

    def reset(self, init_seed=0):
        if init_seed != 0:
            observation, info = self.env.reset(seed=init_seed)
        else:
            observation, info = self.env.reset()
        obs = self.convert_gym_obs_to_agent_obs(observation)
        return obs, 0, False

    def convert_gym_obs_to_agent_obs(self, observation):
        """
        Convert the observation from the gym environment to the format expected by the agent.
        """
        self_value, dealer_value, usable_ace = observation
        return self.agent.obs2feature([self_value, usable_ace, dealer_value])

    def step(self, action):
        """
        action: {0,1}
        0: stick
        1: hit
        """
        observation, reward, done, _, _ = self.env.step(action)
        obs = self.convert_gym_obs_to_agent_obs(observation)
        return obs, reward, done
