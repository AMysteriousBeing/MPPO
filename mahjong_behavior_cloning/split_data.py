import numpy as np
import json
import os


def saveData(matchid, lines, split_path):
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    with open(os.path.join(split_path, f"{matchid}.txt"), "w") as f:
        f.writelines(lines)


def split_data(text_path, split_path):
    lines = []
    matchid = -1
    debug = False
    file_name = text_path
    if debug:
        file_name = "data/sample.txt"
    with open(file_name, encoding="UTF-8") as f:
        line = f.readline()
        while line:
            lines.append(line)
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            if t[0] == "Match":
                matchid += 1
            if t[0] == "Score":
                saveData(matchid, lines, split_path)
                lines = []
            line = f.readline()


if __name__ == "__main__":
    text_path = "../data/hzddBot/hzddBot.txt"
    split_path = "../data/hzddBot/split_data/"
    split_data(text_path, split_path)
