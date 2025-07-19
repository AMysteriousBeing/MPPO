from MahJong_feature_agent2_adapted import FeatureFull as FeatureAgent
import numpy as np
import json
import os
from multiprocessing import Pool

# compared to preprocess.py, preprocess_featurefull uses featurefull agent for obs features

TILE_LIST = [
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


def filterData(obs, actions):
    newobs = [[] for i in range(4)]
    newactions = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            if (
                o["action_mask"].sum() > 1
            ):  # ignore states with single valid action (Pass)
                newobs[i].append(o)
                newactions[i].append(actions[i][j])
    obs = newobs
    actions = newactions
    return obs, actions


def processObs(obs, tile_coding):
    newobs = [[] for i in range(4)]
    for i in range(4):
        for j, o in enumerate(obs[i]):
            new_o = {}
            new_o["action_mask"] = o["action_mask"]
            new_o["observation"] = FeatureAgent.feature_normal_from_normal(
                o["observation"], tile_coding
            )
            newobs[i].append(new_o)
    obs = newobs
    return obs


def saveData(sl_data_path, obs, actions, mark, matchid, augment_factor=8):
    assert [len(x) for x in obs] == [
        len(x) for x in actions
    ], "obs actions not matching!"
    sl_true_data_path = os.path.join(sl_data_path, "sl_data_featurefull2/")
    sl_meta_data_path = os.path.join(sl_data_path, "sl_meta_featurefull2/")
    if not os.path.exists(sl_true_data_path):
        os.makedirs(sl_true_data_path)
    if not os.path.exists(sl_meta_data_path):
        os.makedirs(sl_meta_data_path)
    np.savez(
        os.path.join(sl_true_data_path, "%d.npz" % (matchid * augment_factor + mark)),
        obs=np.stack([x["observation"] for i in range(4) for x in obs[i]]).astype(
            np.int8
        ),
        mask=np.stack([x["action_mask"] for i in range(4) for x in obs[i]]).astype(
            np.int8
        ),
        act=np.array([x for i in range(4) for x in actions[i]]),
    )
    with open(
        os.path.join(
            sl_meta_data_path, "{}.json".format(matchid * augment_factor + mark)
        ),
        "w",
    ) as f:
        json.dump(sum([len(x) for x in obs]), f)


def preprocess(split_data_path, sl_data_path, match_id, tile_coding, augment_factor=8):
    obs = [[] for i in range(4)]
    actions = [[] for i in range(4)]
    # for x in obs:
    #     x.clear()
    # for x in actions:
    #     x.clear()
    if os.path.exists(
        os.path.join(
            sl_data_path,
            "sl_data_featurefull2",
            "{}.npz".format(match_id * augment_factor + tile_coding),
        )
    ) and os.path.exists(
        os.path.join(
            sl_data_path,
            "sl_meta_featurefull2",
            "{}.json".format(match_id * augment_factor + tile_coding),
        )
    ):
        return

    with open(
        os.path.join(split_data_path, "{}.txt".format(match_id)), encoding="UTF-8"
    ) as f:
        line = f.readline()
        while line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            if t[0] == "Match":
                agents = [FeatureAgent(i, TILE_LIST[tile_coding]) for i in range(4)]
            elif t[0] == "Wind":
                for agent in agents:
                    agent.request2obs(line)
            elif t[0] == "Player":
                p = int(t[1])
                if t[2] == "Deal":
                    agents[p].request2obs(" ".join(t[2:]))
                elif t[2] == "Draw":
                    for i in range(4):
                        if i == p:
                            obs[p].append(agents[p].request2obs(" ".join(t[2:])))
                            actions[p].append(0)
                        else:
                            agents[i].request2obs(" ".join(t[:3]))
                elif t[2] == "Play":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action(" ".join(t[2:])))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs(line)
                        else:
                            obs[i].append(agents[i].request2obs(line))
                            actions[i].append(0)
                    curTile = t[3]
                elif t[2] == "Chi":
                    actions[p].pop()
                    actions[p].append(
                        agents[p].response2action("Chi %s %s" % (curTile, t[3]))
                    )
                    for i in range(4):
                        if i == p:
                            obs[p].append(
                                agents[p].request2obs("Player %d Chi %s" % (p, t[3]))
                            )
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
                elif t[2] == "Peng":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            obs[p].append(
                                agents[p].request2obs("Player %d Peng %s" % (p, t[3]))
                            )
                            actions[p].append(0)
                        else:
                            agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
                elif t[2] == "Gang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("Gang %s" % t[3]))
                    for i in range(4):
                        agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
                elif t[2] == "AnGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                        else:
                            agents[i].request2obs("Player %d AnGang" % p)
                elif t[2] == "BuGang":
                    actions[p].pop()
                    actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                    for i in range(4):
                        if i == p:
                            agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                        else:
                            obs[i].append(
                                agents[i].request2obs("Player %d BuGang %s" % (p, t[3]))
                            )
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
                obs, actions = filterData(obs, actions)
                obs = processObs(obs, TILE_LIST[tile_coding])
                saveData(
                    sl_data_path,
                    obs,
                    actions,
                    tile_coding,
                    match_id,
                    augment_factor,
                )
            line = f.readline()


if __name__ == "__main__":

    split_data_path = "../data/hzddBot/split_data/"
    sl_data_path = "../data/hzddBot/"
    augmentation_times = 2
    num_workers = os.cpu_count() // 2
    pool = Pool(num_workers)
    for code in range(augmentation_times):
        file_list = os.listdir(split_data_path)
        for i in range(32768):
            match_id = i
            ret = pool.apply_async(
                preprocess,
                args=(
                    split_data_path,
                    sl_data_path,
                    match_id,
                    code,
                    augmentation_times,
                ),
            )
            # preprocess(
            #     split_data_path, sl_data_path, match_id, code, augmentation_times
            # )

    pool.close()
    pool.join()
    sl_meta_data_path = os.path.join(sl_data_path, "sl_meta_featurefull2/")
    file_list = os.listdir(sl_meta_data_path)
    data_length_list = []
    for i in range(len(file_list)):
        with open(os.path.join(sl_meta_data_path, "{}.json".format(i)), "r") as f:
            data_length = json.load(f)
            data_length_list.append(data_length)

    with open(os.path.join(sl_data_path, "count_featurefull2.json"), "w") as f:
        json.dump(data_length_list, f)
