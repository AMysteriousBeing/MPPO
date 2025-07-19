from agent.Maze_agent_template import MazeAgentTemplate
import torch
import numpy as np


def visualize_data(data):
    for row in data:
        for cell in row:
            if cell == -2:
                print("~", end=" ")
            elif cell == -1:
                print("█", end=" ")
            elif 0 < cell:
                print("X", end=" ")
            else:
                print("`", end=" ")
        print()


class MazeAgentPassThrough(MazeAgentTemplate):

    def __init__(self, view_size, maze_size):
        self.view_size = view_size
        self.maze_size = maze_size

    def obs2feature(self, obs):
        cnn_obs = obs["cnn_obs"][0]
        dense_obs = obs["dense_obs"]

        return {
            "cnn_obs": cnn_obs,
            "dense_obs": dense_obs,
            "action_mask": self.action_mask(obs),
        }

    def action2response(self, action):
        return action

    def action_mask(self, obs):
        """
        1 if the action is valid, 0 if not
        action = 0,1,2,3 for up, down, left, right
        """
        cnn_feature = obs["cnn_obs"][0][
            self.view_size - 1 : self.view_size + 2,
            self.view_size - 1 : self.view_size + 2,
        ]
        action_mask = torch.zeros(4)
        if cnn_feature[0, 1] == 0:
            action_mask[0] = 1
        if cnn_feature[1, 2] == 0:
            action_mask[3] = 1
        if cnn_feature[1, 0] == 0:
            action_mask[2] = 1
        if cnn_feature[2, 1] == 0:
            action_mask[1] = 1
        return action_mask
