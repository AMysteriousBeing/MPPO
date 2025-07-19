import numpy as np
import json
import copy

CANONICAL_TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]
CANONICAL_OFFSET_TILE = {c: i for i, c in enumerate(CANONICAL_TILE_LIST)}
CANONICAL_OFFSET_ACT = {
    "Pass": 0,
    "Hu": 1,
    "Play": 2,
    "Chi": 36,
    "Peng": 99,
    "Gang": 133,
    "AnGang": 167,
    "BuGang": 201,
}
CANONICAL_OFFSET_ACT2 = {
    "Pass": 0,
    "Hu": 1,
    "Play": 2,
    "Chi": 36,
    "Peng": 99,
    "Gang": 133,
    "BuGang": 167,
    "AnGang": 201,
}
AUGMENTATION_SCHEME = [
    [
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("B%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i) % 3 + 1) for i in range(3)),
    ],
    [
        *("W%d" % (i + 1) for i in range(9)),
        *("B%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ],
    [
        *("T%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("B%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 2) % 3 + 1) for i in range(3)),
    ],
    [
        *("T%d" % (i + 1) for i in range(9)),
        *("B%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 3) % 3 + 1) for i in range(3)),
    ],
    [
        *("B%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ],
    [
        *("B%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 2) % 3 + 1) for i in range(3)),
    ],
    [
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 3) % 3 + 1) for i in range(3)),
    ],
    [
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ],
    [
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 2) % 3 + 1) for i in range(3)),
    ],
    [
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 3) % 3 + 1) for i in range(3)),
    ],
    [
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ],
    [
        *("B%d" % (i + 1) for i in range(8, -1, -1)),
        *("W%d" % (i + 1) for i in range(8, -1, -1)),
        *("T%d" % (i + 1) for i in range(8, -1, -1)),
        *("F%d" % ((i) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 2) % 3 + 1) for i in range(3)),
    ],
]


def data_augmentation(line, scheme_id):
    """
    For each tile, convert to 34 representation, then back to tile
    """
    ret_line = []
    for word in line.split():
        if word in CANONICAL_TILE_LIST:
            tile_offset = CANONICAL_OFFSET_TILE[word]
            word = AUGMENTATION_SCHEME[scheme_id][tile_offset]
        ret_line.append(word)
    return " ".join(ret_line)


def init_data_augmentation_SIL(init_data, scheme_id):
    """
    Augment "tileWall" in init_data according to scheme_id
    Return a copy of init_data
    """
    init_data_copy = copy.deepcopy(init_data)
    # tile_wall is a list of list
    tile_wall = init_data_copy["tileWall"]

    augmented_tile_wall = []
    for single_tile_wall in tile_wall:
        augmented_single_tile_wall = []
        for tile in single_tile_wall:
            tile_offset = CANONICAL_OFFSET_TILE[tile]
            augmented_tile = AUGMENTATION_SCHEME[scheme_id][tile_offset]
            augmented_single_tile_wall.append(augmented_tile)
        augmented_tile_wall.append(augmented_single_tile_wall)
    init_data_copy["tileWall"] = augmented_tile_wall
    return init_data_copy


def action2response(action):
    if action < CANONICAL_OFFSET_ACT["Hu"]:
        return "Pass"
    if action < CANONICAL_OFFSET_ACT["Play"]:
        return "Hu"
    if action < CANONICAL_OFFSET_ACT["Chi"]:
        return "Play " + CANONICAL_TILE_LIST[action - CANONICAL_OFFSET_ACT["Play"]]
    if action < CANONICAL_OFFSET_ACT["Peng"]:
        t = (action - CANONICAL_OFFSET_ACT["Chi"]) // 3
        return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
    if action < CANONICAL_OFFSET_ACT["Gang"]:
        return "Peng"
    if action < CANONICAL_OFFSET_ACT["AnGang"]:
        return "Gang"
    if action < CANONICAL_OFFSET_ACT["BuGang"]:
        return "Gang " + CANONICAL_TILE_LIST[action - CANONICAL_OFFSET_ACT["AnGang"]]
    return "BuGang " + CANONICAL_TILE_LIST[action - CANONICAL_OFFSET_ACT["BuGang"]]


class DummyAgent:
    def __init__(self, version=1):
        if version == 1:
            self.offset_act = CANONICAL_OFFSET_ACT
        elif version == 2:
            self.offset_act = CANONICAL_OFFSET_ACT2

    def response2action(self, response):
        t = response.split()
        if t[0] == "Pass":
            return self.offset_act["Pass"]
        if t[0] == "Hu":
            return self.offset_act["Hu"]
        if t[0] == "Play":
            return self.offset_act["Play"] + CANONICAL_OFFSET_TILE[t[1]]
        if t[0] == "Chi":
            return (
                self.offset_act["Chi"]
                + "WTB".index(t[1][0]) * 7 * 3
                + (int(t[2][1]) - 2) * 3
                + int(t[1][1])
                - int(t[2][1])
                + 1
            )
        if t[0] == "Peng":
            return self.offset_act["Peng"] + CANONICAL_OFFSET_TILE[t[1]]
        if t[0] == "Gang":
            return self.offset_act["Gang"] + CANONICAL_OFFSET_TILE[t[1]]
        if t[0] == "BuGang":
            return self.offset_act["BuGang"] + CANONICAL_OFFSET_TILE[t[1]]
        if t[0] == "AnGang":
            return self.offset_act["AnGang"] + CANONICAL_OFFSET_TILE[t[1]]
        return self.offset_act["Pass"]


def response2action_SIL(response):
    t = response.split()
    if t[0] == "Pass":
        return CANONICAL_OFFSET_ACT["Pass"]
    if t[0] == "Hu":
        return CANONICAL_OFFSET_ACT["Hu"]
    if t[0] == "Play":
        return CANONICAL_OFFSET_ACT["Play"] + CANONICAL_OFFSET_TILE[t[1]]
    if t[0] == "Chi":
        return (
            CANONICAL_OFFSET_ACT["Chi"]
            + "WTB".index(t[1][0]) * 7 * 3
            + (int(t[2][1]) - 2) * 3
            + int(t[1][1])
            - int(t[2][1])
            + 1
        )
    if t[0] == "Peng":
        return CANONICAL_OFFSET_ACT["Peng"] + CANONICAL_OFFSET_TILE[t[1]]
    if t[0] == "Gang":
        return CANONICAL_OFFSET_ACT["Gang"] + CANONICAL_OFFSET_TILE[t[1]]
    if t[0] == "BuGang":
        return CANONICAL_OFFSET_ACT["BuGang"] + CANONICAL_OFFSET_TILE[t[1]]
    if t[0] == "AnGang":
        return CANONICAL_OFFSET_ACT["AnGang"] + CANONICAL_OFFSET_TILE[t[1]]
    return CANONICAL_OFFSET_ACT["Pass"]


def load_actions(bz_log, scheme_id, version=1):
    """
    Note: Force scheme_id = 0 for 推不倒 and 绿一色
    """
    actions = [[] for i in range(4)]
    for x in actions:
        x.clear()
    line_id = 0
    for line in bz_log:
        line = data_augmentation(line, scheme_id)
        # print(line)
        # print(line_id, line)
        t = line.split()
        if len(t) == 0:
            continue
        if t[0] == "Match":
            agents = [DummyAgent(version) for _ in range(4)]
        elif t[0] == "Wind":
            pass
        elif t[0] == "Player":
            p = int(t[1])
            if t[2] == "Deal":
                pass
            elif t[2] == "Draw":
                for i in range(4):
                    if i == p:
                        actions[p].append(0)
            elif t[2] == "Play":
                actions[p].pop()
                actions[p].append(agents[p].response2action(" ".join(t[2:])))
                for i in range(4):
                    if i != p:
                        actions[i].append(0)
                curTile = t[3]
            elif t[2] == "Chi":
                actions[p].pop()
                actions[p].append(
                    agents[p].response2action("Chi %s %s" % (curTile, t[3]))
                )
                for i in range(4):
                    if i == p:
                        actions[p].append(0)
            elif t[2] == "Peng":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                for i in range(4):
                    if i == p:
                        actions[p].append(0)
            elif t[2] == "Gang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Gang %s" % t[3]))
            elif t[2] == "AnGang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
            elif t[2] == "BuGang":
                actions[p].pop()
                actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                for i in range(4):
                    if i != p:
                        actions[i].append(0)
            elif t[2] == "Hu":
                actions[p].pop()
                actions[p].append(agents[p].response2action("Hu"))
            # Deal with Ignore clause
            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == "Chi":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action(
                                    "Chi %s %s" % (curTile, t[k + 3])
                                )
                            )
                        elif t[k + 2] == "Peng":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action("Peng %s" % t[k + 3])
                            )
                        elif t[k + 2] == "Gang":
                            actions[p].pop()
                            actions[p].append(
                                agents[p].response2action("Gang %s" % t[k + 3])
                            )
                        elif t[k + 2] == "Hu":
                            actions[p].pop()
                            actions[p].append(agents[p].response2action("Hu"))
                    else:
                        break
        elif t[0] == "Score":
            pass
        line_id += 1
    return actions


if __name__ == "__main__":
    file_path = "../data_pass2/1.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    actions = load_actions(data["history"], 0)
    # ag = DummyAgent()
    # a = ag.response2action("Play F2")
    print(actions)
