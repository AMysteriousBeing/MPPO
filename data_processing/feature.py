from collections import defaultdict
import numpy as np
import sys
import copy
import os

# sys.path.append(os.path.join(os.getcwd(), ".."))
# sys.path.append(os.getcwd())

# try:
#     from MahjongGB import MahjongFanCalculator
# except:
#     print(
#         "MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information."
#     )
#     raise


def default_value():
    return 0


# fmt: off
tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
            'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
            ]
# fmt: on


def load_log(path, name):
    """
    load log and read data
    """
    dst_path = os.path.join(path, name)
    with open(dst_path, "rb") as f:
        botzone_log = np.load(f, allow_pickle=True)
        tileWall_log = np.load(f, allow_pickle=True)
        pack_log = np.load(f, allow_pickle=True)
        handWall_log = np.load(f, allow_pickle=True)
        obsWall_log = np.load(f, allow_pickle=True)
        remaining_tile_log = np.load(f, allow_pickle=True)

        # meta
        botzone_id = np.load(f, allow_pickle=True)[0]
        winner_id = np.load(f, allow_pickle=True)[0]
        prevalingWind = np.load(f, allow_pickle=True)[0]
        fan_sum = np.load(f, allow_pickle=True)[0]
        score = np.load(f, allow_pickle=True)
        fan_list = np.load(f, allow_pickle=True)
    return (
        botzone_log,  # 1
        tileWall_log,  # 2
        pack_log,  # 3
        handWall_log,  # 4
        obsWall_log,  # 5
        remaining_tile_log,  # 6
        botzone_id,  # 7
        winner_id,  # 8
        prevalingWind,  # 9
        fan_sum,  # 10
        score,
        fan_list,  # 11
    )


class DataFormatting:
    def __init__(self, bz_match_id=0):
        """
        Initialize a 4-player perspective, in which pack is "附录",
        tileWall is remaining tiles from the persepctive of players, handWall is private hand tiles of players
        tileWall is initialized to full Chinese Standard Mahjong tiles(34*4)
        """
        self.botzone_match_id = bz_match_id

        self.seatWind = [0, 1, 2, 3]
        # pack, tileWall, handWall, from player x's perspective
        self.packs = [[] for i in range(4)]
        self.tileWall = [defaultdict(default_value) for _ in range(4)]
        self.handWall = [defaultdict(default_value) for _ in range(4)]

        # tile wall, from public view, excluding players' secret hand wall
        self.tileWall_obs = [defaultdict(default_value)]

        self.played_tile = None
        self.drawn_tile = None
        self.last_play_id = -1
        self.init_tileWall()

        # game log
        self.pack_log = []
        self.tileWall_log = []
        self.handWall_log = []
        self.obsWall_log = []
        self.botzone_log = []
        self.game_won_id = -1
        self.prevalingWind = -1
        self.fan_sum = -1
        self.remaining_tile = [21, 21, 21, 21]
        self.remaining_tile_log = []
        self.fan_list = []
        self.score = []

    def update_history(self, log):
        """
        push updated info to game history
        """
        self.pack_log.append(copy.deepcopy(self.packs))
        self.tileWall_log.append(copy.deepcopy(self.tileWall))
        self.handWall_log.append(copy.deepcopy(self.handWall))
        self.obsWall_log.append(copy.deepcopy(self.tileWall_obs))
        self.botzone_log.append(log)
        self.remaining_tile_log.append(copy.deepcopy(self.remaining_tile))

    def trial(self):
        """
        for quick code testing
        """
        # for i in range(4):
        #     print("Player {}".format(i))
        #     print(self.tileWall[i])
        #     print(self.handWall[i])
        #     print(self.packs[i])
        #     print("")
        # print(self.remaining_tile_log)
        return

    def save_log(self, path, name):
        """
        save game log data to file
        """
        if not os.path.exists(path):
            os.makedirs(path)
        dst_path = os.path.join(path, name)
        with open(dst_path, "wb") as f:

            np.save(f, np.array(self.botzone_log, dtype=object))
            np.save(f, np.array(self.tileWall_log, dtype=object))
            np.save(f, np.array(self.pack_log, dtype=object))
            np.save(f, np.array(self.handWall_log, dtype=object))
            np.save(f, np.array(self.obsWall_log, dtype=object))
            np.save(f, np.array(self.remaining_tile_log, dtype=object))

            # meta
            np.save(f, [self.botzone_match_id])
            np.save(f, [self.game_won_id])
            np.save(f, [self.prevalingWind])
            np.save(f, [self.fan_sum])
            np.save(f, self.score)
            np.save(f, self.fan_list)

    def init_tileWall(self):
        """
        init tileWall to full
        """
        for tile in tile_list_raw:
            self.tileWall[0][tile] = 4
            self.tileWall[1][tile] = 4
            self.tileWall[2][tile] = 4
            self.tileWall[3][tile] = 4
            self.tileWall_obs[0][tile] = 4

    def update_obsWall(self, tile_list):
        """
        Update public viewable tile
        tile_list: canonical encoding, tiles to subtract
        """
        for tile in tile_list:
            if self.tileWall_obs[0][tile] != 0:
                self.tileWall_obs[0][tile] -= 1
            else:
                print("### ERROR ###")
                print("count is: ", self.tileWall[0][tile])
                print("tileWall_obs, tile: {}".format(tile))
                raise RuntimeError("Inconsistent tileWall_obs")

    def update_tileWall(self, player_id, tile_list):
        """
        Update tileWall for player_id's perspective
        Reduce tiles_list from tileWall
        tile_list: canonical encoding
        """
        for tile in tile_list:

            if self.tileWall[player_id][tile] != 0:
                self.tileWall[player_id][tile] -= 1
            else:
                print("### ERROR ###")
                print("count is: ", self.tileWall[player_id][tile])
                print("player_id: {}, tile: {}".format(player_id, tile))
                raise RuntimeError("Inconsistent tileWall")

    def update_handWall(self, player_id, tile_list_addition, tile_list_subtraction):
        """
        Update tileWall for player_id's perspective
        tile_list_*: canonical encoding
        """
        for tile in tile_list_subtraction:
            if self.handWall[player_id][tile] != 0:
                self.handWall[player_id][tile] -= 1
                if self.handWall[player_id][tile] == 0:
                    del self.handWall[player_id][tile]
            else:
                raise RuntimeError("Inconsistent tileWall")
        for tile in tile_list_addition:
            self.handWall[player_id][tile] += 1

    def update_pack(self, player_id, tile_list, is_AnGang=False):
        """
        Update tileWall for player_id's perspective
        Add tile_list to pack
        tile_list: canonical encoding
        """
        temp_dict = defaultdict(default_value)
        for tile in tile_list:
            temp_dict[tile] += 1
        if is_AnGang:
            temp_dict["AnGang"] = True
        self.packs[player_id].append(temp_dict)

    def pack_elevation(self, player_id, tile):
        """
        Update pack: elevation peng to gang for BuGang operation for "tile"
        """
        for entry in self.packs[player_id]:
            if tile in entry and entry[tile] == 3:
                entry[tile] = 4
                return
        raise RuntimeError("Invalid Pack Elevation Operation")

    def request2obs(self, request):
        """
        transform botzone style Mahjong game log to round-specific data format
        """
        t = request.split()
        if t[0] == "Wind":
            self.prevalingWind = int(t[1])
            self.update_history(request)
            return

        if t[0] == "Huang":
            self.update_history(request)
            return

        if t[2] == "Deal":
            player_id = int(t[1])
            self.update_tileWall(player_id, t[3:])
            self.update_handWall(player_id, t[3:], [])
            self.update_history(request)
            return

        if t[2] == "Draw":
            # Available: Hu, Play, AnGang, BuGang
            player_id = int(t[1])
            self.last_play_id = player_id
            self.remaining_tile[player_id] -= 1
            self.drawn_tile = t[3]
            self.last_play_id = player_id
            self.update_tileWall(player_id, [t[3]])
            self.update_handWall(player_id, [t[3]], [])
            tile = t[1]
            self.update_history(request)
            return

        if t[2] == "Invalid":
            return

        if t[2] == "Hu":
            player_id = int(t[1])
            # deal with error in 自摸
            if player_id == self.last_play_id:
                self.game_won_id = player_id
                t[3] = self.drawn_tile
                self.update_history(" ".join(t))
            else:
                self.update_handWall(player_id, [t[3]], [])
                self.game_won_id = player_id
                self.update_history(request)
            return

        if t[2] == "Play":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # for specific player, update handWall
            self.update_handWall(player_id, [], [t[3]])
            self.played_tile = t[3]
            # for other players, update tileWall
            for i in range(4):
                if i != player_id:
                    self.update_tileWall(i, [t[3]])
            # update public view
            self.update_obsWall([t[3]])
            # update history
            self.update_history(request)
            return
        if t[2] == "Chi":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # update pack
            tile_type = t[3][0]
            tile_rank = int(t[3][1])
            tile = t[3]
            tile_higher = tile_type + str(tile_rank + 1)
            tile_lower = tile_type + str(tile_rank - 1)
            chi_composition = [tile_lower, tile, tile_higher]
            self.update_pack(player_id, chi_composition)
            # update handWall
            hand_usage = []
            for i in chi_composition:
                if i != self.played_tile:
                    hand_usage.append(i)
            self.update_handWall(player_id, [], hand_usage)

            # update other player's tile_wall
            for i in range(4):
                if i != player_id:
                    self.update_tileWall(i, hand_usage)

            # update public view
            self.update_obsWall(hand_usage)
            # update history
            self.update_history(request)
            return
        if t[2] == "Peng":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # update pack
            tile = t[3]
            peng_composition = [tile, tile, tile]
            self.update_pack(player_id, peng_composition)
            # update handWall
            hand_usage = [tile, tile]
            self.update_handWall(player_id, [], hand_usage)

            # update other player's tile_wall
            for i in range(4):
                if i != player_id:
                    self.update_tileWall(i, hand_usage)

            # update public view
            self.update_obsWall(hand_usage)
            # update history
            self.update_history(request)
            return
        if t[2] == "Gang":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # update pack
            tile = t[3]
            gang_composition = [tile, tile, tile, tile]
            self.update_pack(player_id, gang_composition)
            # update handWall
            hand_usage = [tile, tile, tile]
            self.update_handWall(player_id, [], hand_usage)

            # update other player's tile_wall
            for i in range(4):
                if i != player_id:
                    self.update_tileWall(i, hand_usage)
            # update public view
            self.update_obsWall(hand_usage)
            # update history
            self.update_history(request)
            return
        if t[2] == "AnGang":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # update pack
            tile = t[3]
            gang_composition = [tile, tile, tile, tile]
            self.update_pack(player_id, gang_composition, is_AnGang=True)
            # update handWall
            self.update_handWall(player_id, [], gang_composition)

            # DO NOT update other player's tile_wall
            self.update_history(request)
            return
        if t[2] == "BuGang":
            # specify player_id
            player_id = int(t[1])
            self.last_play_id = player_id
            # update pack
            tile = t[3]
            self.pack_elevation(player_id, tile)
            # update handWall
            hand_usage = [tile]
            self.update_handWall(player_id, [], hand_usage)

            # update other player's tile_wall
            for i in range(4):
                if i != player_id:
                    self.update_tileWall(i, hand_usage)
            # update public view
            self.update_obsWall(hand_usage)
            # update history
            self.update_history(request)
            return
        if t[0] == "Fan":
            # fan_sum: 番数总和
            # fan_style: [番种*x]
            # fan_list: [番种，番种， ...]
            self.fan_sum = int(t[1])
            fan_style = t[2].split("+")
            for fan in fan_style:
                fan_detail = fan.split("*")
                for _ in range(int(fan_detail[1])):
                    self.fan_list.append(fan_detail[0])

            return
        if t[0] == "Score":
            # fan_sum: 番数总和
            # fan_style: [番种*x]
            # fan_list: [番种，番种， ...]
            self.score = list(map(int, t[1:]))
            return
        raise NotImplementedError("Unknown request %s!" % request)
