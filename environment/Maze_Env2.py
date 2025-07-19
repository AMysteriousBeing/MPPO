import random
import numpy as np
from environment.Maze_Gen2 import MazeGenerator2
from agent import MazeAgentTemplate


def calc_distance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class MazeEnv2:
    """
    Partial Observable Maze Environment of 19*19 with a single agent.
    The agent receives a reward of 3 for reaching the end location,
    -0.01 for each step taken, and 0 for losing.
    The maze is generated randomly with a seed for reproducibility.
    Observation Space: 7x7 grid around the agent's current position.
    Action Space: 4 discrete actions (up, down, left, right).
    The maze is represented as a 2D numpy array where:
    -1 represents walls,
    0 represents empty spaces,
    """

    def __init__(self, config):
        self.win_reward = 1.5
        self.step_reward = 0  # -0.02
        self.dense_reward = 0  # 0.003
        self.lose_reward = 0
        self.truncate_step = 80
        self.view_size = 5
        self.maze_size = 19
        self.revisit_penalty = 0  # -0.002

        self.maze_dim_config = (self.maze_size - 1) // 2

        self.obs = []
        assert "agent_type" in config, "must specify agent_type to process features!"
        self.agent_type = config["agent_type"]
        assert issubclass(
            self.agent_type, MazeAgentTemplate
        ), "ageng_type must be a subclass of MazeAgentTemplate!"
        self.agent = self.agent_type(self.view_size, self.maze_size)
        self.debug = config.get("debug", False)
        self.oracle_info = config.get("oracle", False)
        self.agent_processing = config.get("agent_processing", True)

    def _generate_maze_with_buffer(self):
        self.buffered_maze = np.zeros(
            (self.maze_size + self.view_size * 2, self.maze_size + self.view_size * 2),
            dtype=np.float32,
        )
        self.buffered_visited_map = np.zeros(
            (self.maze_size + self.view_size * 2, self.maze_size + self.view_size * 2),
            dtype=np.float32,
        )
        self.buffered_maze[:, :] = -2
        self.buffered_maze[
            self.view_size : self.maze_size + self.view_size,
            self.view_size : self.maze_size + self.view_size,
        ] = self.maze
        self.buffered_visited_map[
            self.view_size : self.maze_size + self.view_size,
            self.view_size : self.maze_size + self.view_size,
        ] = self.visited_map

    def reset(self, init_seed=None):
        self.step_count = 0
        self.seed = init_seed
        self.maze_gen = MazeGenerator2(self.maze_dim_config)
        self.maze_gen.generate_maze(seed=self.seed)
        self.maze = self.maze_gen.get_maze()
        self.location = self.maze_gen.get_start_location()
        self.prev_location = self.location
        self.prev_prev_location = self.location
        self.start_pos = self.maze_gen.get_start_location()
        self.end_pos = self.maze_gen.get_end_location()
        self.visited_map = (
            np.ones(
                (self.maze_size, self.maze_size),
                dtype=np.float32,
            )
            * -1
        )
        self.visited_map[self.location] += 1
        self._generate_maze_with_buffer()
        self._update_obs()
        if self.debug:
            self.visualize()
            print(f"Start Position: {self.start_pos}, End Position: {self.end_pos}")

        if self.agent_processing:
            self.obs = self.agent.obs2feature(
                {"cnn_obs": self.cnn_obs, "dense_obs": self.dense_obs}
            )
            return self.obs, 0, False
        else:
            return ({"cnn_obs": self.cnn_obs, "dense_obs": self.dense_obs}, 0, False)

    def expose_obs(self):
        """
        Expose the current observation.
        """
        return {
            "cnn_obs": self.cnn_obs,
            "dense_obs": self.dense_obs,
        }

    def get_reference_action(self):
        """
        return action according to maze's solution
        """
        solution = self.maze_gen.find_path(self.location)
        next_position = solution[0]
        if next_position[0] > self.location[0]:
            # going down
            return 1
        if next_position[0] < self.location[0]:
            return 0
        if next_position[1] > self.location[1]:
            return 3
        if next_position[1] < self.location[1]:
            return 2
        # fall back
        return -1

    def _update_obs(self):

        self.observable_obs = self.buffered_maze[
            self.location[0] : self.location[0] + self.view_size * 2 + 1,
            self.location[1] : self.location[1] + self.view_size * 2 + 1,
        ]
        self.revisit_obs = self.buffered_visited_map[
            self.location[0] : self.location[0] + self.view_size * 2 + 1,
            self.location[1] : self.location[1] + self.view_size * 2 + 1,
        ]
        self.cnn_obs = np.stack([self.observable_obs, self.revisit_obs])
        solution = self.maze_gen.find_path(self.location)
        if solution == None or not self.oracle_info:
            len_solution = 0
        else:
            len_solution = len(solution)
        # dense features
        self.dense_obs = [
            self.location[0] / self.maze_size,
            self.location[1] / self.maze_size,
            self.prev_location[0] / self.maze_size,
            self.prev_location[1] / self.maze_size,
            self.prev_prev_location[0] / self.maze_size,
            self.prev_prev_location[1] / self.maze_size,
            self.step_count / self.truncate_step,
            self.visited_map[self.location] / 5,
            self.visited_map[self.prev_location] / 5,
            self.visited_map[self.prev_prev_location] / 5,
            len_solution / 100.0,
        ]

    def step(self, action):
        """
        action = 0,1,2,3 for up, down, left, right
        fast forward to turns or forks
        """
        if action == 0:
            dx, dy = -1, 0
        elif action == 1:
            dx, dy = 1, 0
        elif action == 2:
            dx, dy = 0, -1
        elif action == 3:
            dx, dy = 0, 1
        else:
            dx, dy = 0, 0
        self.prev_prev_location = self.prev_location
        self.prev_location = self.location
        tmp_loc = None
        new_loc = self.location
        while new_loc != tmp_loc:
            tmp_loc = new_loc
            # handle illegal moves
            if (
                0 > tmp_loc[0] + dx
                or tmp_loc[0] + dx >= self.maze_size
                or 0 > tmp_loc[1] + dy
                or tmp_loc[1] + dy >= self.maze_size
                or self.maze[tmp_loc[0] + dx, tmp_loc[1] + dy] < 0
            ):
                dx, dy = 0, 0
            # update location
            new_loc = (tmp_loc[0] + dx, tmp_loc[1] + dy)

            # check for fork paths
            path_count = 0
            for test_x, test_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_x, new_y = new_loc[0] + test_x, new_loc[1] + test_y
                if (
                    0 <= new_x < self.maze_size
                    and 0 <= new_y < self.maze_size
                    and self.maze[new_x, new_y] >= 0
                ):
                    path_count += 1
            if path_count > 2:
                break
        self.location = new_loc
        self.visited_map[self.location] += 1
        self.visited_map[self.location] = max(5, self.visited_map[self.location])
        self._generate_maze_with_buffer()

        self.step_count += 1
        if self.step_count >= self.truncate_step:
            done = True
            reward = self.lose_reward
        else:
            done = False
            reward = self.step_reward

        if self.location == self.end_pos:
            reward = self.win_reward
            done = True
        self._update_obs()
        if self.debug:
            print(
                f"Step: {self.step_count}, Location: {self.location}, Reward: {reward}, Done: {done}"
            )
            self.visualize()
        # adding in dense reward
        cur_dist = calc_distance(self.location, self.end_pos)
        prev_dist = calc_distance(self.prev_location, self.end_pos)
        if cur_dist < prev_dist:
            reward += self.dense_reward
        elif cur_dist > prev_dist:
            reward -= self.dense_reward
        # adding revisiting penalty
        revisit_penalty = (
            max(5, self.visited_map[self.location[0], self.location[1]])
            * self.revisit_penalty
        )
        if self.agent_processing:
            self.obs = self.agent.obs2feature(
                {"cnn_obs": self.cnn_obs, "dense_obs": self.dense_obs}
            )
            return self.obs, reward + revisit_penalty, done
        else:
            return (
                {"cnn_obs": self.cnn_obs, "dense_obs": self.dense_obs},
                reward + revisit_penalty,
                done,
            )

    def visualize(self):
        maze_vis = self.maze.copy()
        maze_vis[self.location[0], self.location[1]] = 1
        for row in maze_vis:
            for cell in row:
                if cell == -1:
                    print("█", end=" ")
                elif 0 < cell:
                    print(int(cell), end=" ")
                else:
                    print(" ", end=" ")
            print()

    def visualize_buffered_maze(self):
        for row in self.buffered_maze:
            for cell in row:
                if cell == -2:
                    print("~", end=" ")
                elif cell == -1:
                    print("█", end=" ")
                elif 0 < cell:
                    print(cell, end=" ")
                else:
                    print(" ", end=" ")
            print()


if __name__ == "__main__":
    # Example usage
    me1 = MazeEnv()
    me1.reset()
    me1.visualize_buffered_maze()
