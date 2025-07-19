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


class MazeAgentBasic(MazeAgentTemplate):

    def __init__(self, view_size, maze_size):
        self.view_size = view_size
        self.maze_size = maze_size

    def obs2feature(self, obs):
        cnn_obs = obs["cnn_obs"][0]
        visited_obs = obs["cnn_obs"][1]
        dense_obs = obs["dense_obs"]
        cur_location = (dense_obs[0] * self.maze_size, dense_obs[1] * self.maze_size)
        prev_location = (dense_obs[2] * self.maze_size, dense_obs[3] * self.maze_size)
        prev_prev_location = (
            dense_obs[4] * self.maze_size,
            dense_obs[5] * self.maze_size,
        )
        prev_x, prev_y = (
            self.view_size + int(prev_location[0] - cur_location[0]),
            self.view_size + int(prev_location[1] - cur_location[1]),
        )
        prev_x = max(0, min(prev_x, self.view_size * 2))
        prev_y = max(0, min(prev_y, self.view_size * 2))
        prev_prev_x, prev_prev_y = (
            self.view_size + int(prev_prev_location[0] - cur_location[0]),
            self.view_size + int(prev_prev_location[1] - cur_location[1]),
        )
        prev_prev_x = max(0, min(prev_prev_x, self.view_size * 2))
        prev_prev_y = max(0, min(prev_prev_y, self.view_size * 2))

        cnn_feature = np.zeros(
            (5, self.view_size * 2 + 1, self.view_size * 2 + 1), dtype=np.float32
        )
        cnn_feature[0] = cnn_obs < 0
        cnn_feature[1] = cnn_obs >= 0
        cnn_feature[2] = visited_obs / 5
        cnn_feature[3, prev_x, prev_y] = 1
        cnn_feature[4, prev_prev_x, prev_prev_y] = 1
        dense_feature = torch.tensor(dense_obs, dtype=torch.float32)

        return {
            "cnn_obs": torch.tensor(cnn_feature),
            "dense_obs": dense_feature,
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
