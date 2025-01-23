import os
import json
from feature import load_log

# Preprocessing pass 1: exclude Huang games, gather information listed in data_template except shanten_info

UNAUGMENTABLE_FAN_LIST = ["绿一色", "推不倒"]
data_template = {
    "shanten_info": {},
    "botzone_id": "",
    "winner_id": -1,
    "wind": -1,
    "augmentable": True,
    "history": [],
    "initwall": [],
}


def data_preprocess(path_to_data, path_to_support_data, path_to_bz_log, path_to_save):
    """
    segment data into: line-Shanten dict, botzone_id, winner_id, augment-able, history data
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    matchid = -1
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
                (
                    botzone_log,  # 1
                    tileWall_log,  # 2
                    pack_log,  # 3
                    handWall_log,  # 4
                    obsWall_log,  # 5
                    remaining_tile_log,  # 6
                    support_botzone_id,  # 7
                    winner_id,  # 8
                    prevalingWind,  # 9
                    fan_sum,  # 10
                    score,
                    support_fan_list,  # 11
                ) = load_log(path_to_support_data, "{}.npy".format(matchid))
                assert t[1] == support_botzone_id
                json_obj["botzone_id"] = support_botzone_id
                json_obj["winner_id"] = int(winner_id)
                json_obj["wind"] = int(prevalingWind)
                with open(
                    os.path.join(path_to_bz_log, "{}.json".format(support_botzone_id))
                ) as fx:
                    raw_log = json.load(fx)
                init_log = json.loads(raw_log["initdata"])
                # parsed_init_list = init_log["walltiles"].split(" ")
                json_obj["initwall"] = init_log["walltiles"]
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
                    json_obj["history"] = lines
                    path_to_file = os.path.join(path_to_save, "{}.json".format(matchid))
                    with open(path_to_file, "w", encoding="utf-8") as fp:
                        json.dump(json_obj, fp, indent=2)
                # reset for new game
                lines = []
                json_obj = data_template.copy()

            line = f.readline()


if __name__ == "__main__":
    data_preprocess("data.txt", "../data", "../bz_log_raw", "../data_pass1")
