import random
import numpy as np
from collections import deque
from queue import Queue


class MazeGenerator2:
    def __init__(self, dim=9):
        self.dim = dim
        # Create a grid filled with walls
        self.maze = np.ones((dim * 2 + 1, dim * 2 + 1)) * -1  # -1表示墙
        self.start_location = (1, 0)  # 起点位置
        self.end_location = (dim * 2 - 1, dim * 2)  # 终点位置

    def generate_maze(self, seed=None):

        random.seed(seed)
        # Define the starting point
        x, y = (0, 0)
        self.maze = np.ones((self.dim * 2 + 1, self.dim * 2 + 1)) * -1
        self.maze[2 * x + 1, 2 * y + 1] = 0

        # Initialize the stack with the starting point
        stack = [(x, y)]
        while len(stack) > 0:
            x, y = stack[-1]

            # Define possible directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (
                    nx >= 0
                    and ny >= 0
                    and nx < self.dim
                    and ny < self.dim
                    and self.maze[2 * nx + 1, 2 * ny + 1] == -1
                ):
                    self.maze[2 * nx + 1, 2 * ny + 1] = 0
                    self.maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                    stack.append((nx, ny))
                    break
            else:
                stack.pop()

        # Create an entrance and an exit
        self.maze[1, 0] = 0
        self.maze[self.end_location[0], self.end_location[1]] = 0

    def get_maze(self):
        return self.maze

    def get_start_location(self):
        return self.start_location

    def get_end_location(self):
        return self.end_location

    def visualize_maze(self):

        for row in self.maze:
            for cell in row:
                if cell == -1:
                    print("█", end=" ")
                elif 0 < cell:
                    print(cell, end=" ")
                else:
                    print(" ", end=" ")
            print()

    def visualize_maze_with_location(self, location):

        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):

                if i == location[0] and j == location[1]:
                    print("I", end=" ")
                else:
                    if cell == -1:
                        print("█", end=" ")
                    elif 0 < cell:
                        print(cell, end=" ")
                    else:
                        print(" ", end=" ")
            print()

    def get_solution(self):
        # BFS algorithm to find the shortest path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        start = (0, 1)
        end = (self.maze.shape[0] - 2, self.maze.shape[1] - 2)
        visited = np.zeros_like(self.maze, dtype=bool)
        visited[start] = True
        queue = Queue()
        queue.put((start, []))
        while not queue.empty():
            (node, path) = queue.get()
            for dx, dy in directions:
                next_node = (node[0] + dx, node[1] + dy)
                if next_node == end:
                    return path + [next_node]
                if (
                    next_node[0] >= 0
                    and next_node[1] >= 0
                    and next_node[0] < self.maze.shape[0]
                    and next_node[1] < self.maze.shape[1]
                    and self.maze[next_node] == 0
                    and not visited[next_node]
                ):
                    visited[next_node] = True
                    queue.put((next_node, path + [next_node]))
        return None

    def find_path(self, location):
        # BFS algorithm to find the shortest path
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        start = location
        end = (self.maze.shape[0] - 2, self.maze.shape[1] - 2)
        visited = np.zeros_like(self.maze, dtype=bool)
        visited[start] = True
        queue = Queue()
        queue.put((start, []))
        while not queue.empty():
            (node, path) = queue.get()
            for dx, dy in directions:
                next_node = (node[0] + dx, node[1] + dy)
                if next_node == end:
                    return path + [next_node]
                if (
                    next_node[0] >= 0
                    and next_node[1] >= 0
                    and next_node[0] < self.maze.shape[0]
                    and next_node[1] < self.maze.shape[1]
                    and self.maze[next_node] == 0
                    and not visited[next_node]
                ):
                    visited[next_node] = True
                    queue.put((next_node, path + [next_node]))
        return None


if __name__ == "__main__":
    maze_gen = MazeGenerator2(dim=9)
    maze_gen.generate_maze()
    maze_gen.visualize_maze()
    # path = maze_gen.find_path()
    # if path:
    #     print("Path found:", path)
    # else:
    #     print("No path found.")
