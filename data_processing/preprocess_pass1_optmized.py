import os
import json
from feature import load_log

# Preprocessing pass 1: exclude Huang games, gather information listed in data_template except shanten_info

UNAUGMENTABLE_FAN_LIST = ["绿一色", "推不倒"]
data_template = {
    "shanten_info": {
        "0": {
            "6": 6,
        },
        "1": {
            "6": 6,
        },
        "2": {
            "6": 6,
        },
        "3": {
            "6": 6,
        },
    },
    "botzone_id": "",
    "winner_id": -1,
    "wind": -1,
    "augmentable": True,
    "history": [],
    "initwall": [],
}


def data_preprocess(path_to_data, path_to_bz_log, path_to_save):
    """
    segment data into: line-Shanten dict, botzone_id, winner_id, augment-able, history data
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    matchid = -1
    winner_id = -1
    json_obj = data_template.copy()
    lines = []
    with open(path_to_data, encoding="UTF-8") as f:
        line = f.readline()
        while line:
            t = line.split()
            if len(t) == 0:
                line = f.readline()
                continue
            lines.append(line)
            if t[0] == "Match":
                matchid += 1
                if matchid % 1024 == 0:
                    print("Processing match %d %s..." % (matchid, t[1]))

                support_botzone_id = t[1]
                json_obj["botzone_id"] = support_botzone_id
                with open(
                    os.path.join(path_to_bz_log, "{}.json".format(support_botzone_id))
                ) as fx:
                    raw_log = json.load(fx)
                init_log = json.loads(raw_log["initdata"])
                # parsed_init_list = init_log["walltiles"].split(" ")
                json_obj["initwall"] = init_log["walltiles"]
            if t[0] == "Wind":
                prevalingWind = t[1]
                json_obj["wind"] = int(prevalingWind)
            if len(t) > 2 and t[2] == "Hu":
                winner_id = int(t[1])
                json_obj["winner_id"] = winner_id
            if t[0] == "Fan":
                # fan_sum: 番数总和
                # fan_style: [番种*x]
                # fan_list: [番种，番种， ...]
                fan_list = []
                fan_style = t[2].split("+")
                augmentable = True
                for fan in fan_style:
                    fan_detail = fan.split("*")
                    for _ in range(int(fan_detail[1])):
                        fan_list.append(fan_detail[0])
                for fan in UNAUGMENTABLE_FAN_LIST:
                    if fan in fan_list:
                        augmentable = False
                json_obj["augmentable"] = augmentable

            elif t[0] == "Score":
                # save data
                if winner_id != -1:
                    assert int(t[winner_id + 1]) > 0
                    json_obj["history"] = lines
                    path_to_file = os.path.join(path_to_save, "{}.json".format(matchid))
                    with open(path_to_file, "w", encoding="utf-8") as fp:
                        json.dump(json_obj, fp, indent=2)
                # reset for new game
                lines = []
                winner_id = -1
                json_obj = data_template.copy()

            line = f.readline()


if __name__ == "__main__":
    data_preprocess(
        "../data/mahjongNeXt/mahjongNeXt.txt",
        "../data/mahjongNeXt/mahjongNeXt_raw",
        "../data/mahjongNeXt/mahjongNeXt_pass1",
    )
