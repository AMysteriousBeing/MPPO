from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print(
        "MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information."
    )
    raise


def convert_to_fixed_length_binary(number, length):
    if int(number) > 36:
        return [1] * 6
    binary = bin(number)[2:]
    binary_length = len(binary)

    if binary_length < length:
        binary = "0" * (length - binary_length) + binary
    elif binary_length > length:
        binary = binary[binary_length - length :]
    binary = [int(item) for item in binary]
    return binary


class FeatureAgent2Adapted(MahjongGBAgent):

    # quan1+men1+unseen34+hand14+wall10+(history29+meld4*4)*4
    normal_obs_space = (240,)
    # quan1+men1+unseen34+(history29+meld4*4+hand14+wall10)*4
    oracle_obs_space = (312,)
    # unimplemented
    normal_feature_space = (0, 4, 9)
    # unimplemented
    oracle_feature_space = (0, 4, 9)
    # pass1+hu1+play34+chi63+peng34+gang34+angang34+bugang34
    action_space = (235,)

    OFFSET_OBS = {
        "PREVALENT_WIND": 0,
        "SEAT_WIND": 1,
        "UNSHOWN": 2,
        "HAND": 36,
        "WALL": 50,
        "PLAYER_START": 60,
        "PLAYER_LEN": 45,
        "MELD_START": 29,
        "MELD_LEN": 4,
    }
    OFFSET_ACT = {
        "Pass": 0,
        "Hu": 1,
        "Play": 2,
        "Chi": 36,
        "Peng": 99,
        "Gang": 133,
        "AnGang": 167,
        "BuGang": 201,
    }
    # TILE_LIST = [
    #     *("W%d" % (i + 1) for i in range(9)),
    #     *("T%d" % (i + 1) for i in range(9)),
    #     *("B%d" % (i + 1) for i in range(9)),
    #     *("F%d" % (i + 1) for i in range(4)),
    #     *("J%d" % (i + 1) for i in range(3)),
    # ]
    # OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
    # OFFSET_TILE["PUBLIC"] = 34
    # OFFSET_TILE["CONCEALED"] = 35

    def __init__(self, seatWind, tile_coding):
        self.TILE_LIST = tile_coding
        self.OFFSET_TILE = {c: i for i, c in enumerate(self.TILE_LIST)}
        self.OFFSET_TILE["PUBLIC"] = 34
        self.OFFSET_TILE["CONCEALED"] = 35
        self.duplicate = True
        self.seatWind = seatWind
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4 if self.duplicate else 92
        self.wall = []
        self.shownTiles = defaultdict(int)
        self.knownTiles = defaultdict(int)
        self.flower = 0
        self.wallLast = False
        self.myWallLast = False
        self.isAboutKong = False
        self.obs = np.full(self.normal_obs_space, -1, np.uint8)
        self.obs[self.OFFSET_OBS["SEAT_WIND"]] = self.seatWind

    """
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N BuHua
    Player N(me) AnGang XX
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(not me) AnGang
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    """

    def request2obs(self, request):
        t = request.split()
        if t[0] == "Wind":
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS["PREVALENT_WIND"]] = self.prevalentWind
            return
        if t[0] == "Deal":
            self.hand = t[1:]
            self._hand_embedding_update()
            self._unshown_embedding_update()
            return
        if t[0] == "Wall":
            self.wall = t[1:]
            self._wall_embedding_update()
            return
        if t[0] == "Huang":
            self.valid = []
            return self._obs()
        if t[0] == "Draw":
            # Available: Hu, Play, AnGang, BuGang
            if self.duplicate:
                self.tileWall[0] -= 1
                self.wallLast = self.tileWall[1] == 0
                self.myWallLast = self.tileWall[0] == 0
            else:
                self.tileWall -= 1
                self.myWallLast = self.wallLast = self.tileWall == 0
            if self.wall:
                self.wall.pop(0)
                self._wall_embedding_update()
            tile = t[1]
            self.valid = []
            if self._check_mahjong(
                tile, isSelfDrawn=True, isAboutKong=self.isAboutKong
            ):
                self.valid.append(self.OFFSET_ACT["Hu"])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                if (
                    self.hand.count(tile) == 4
                    and not self.wallLast
                    and not self.myWallLast
                ):
                    self.valid.append(
                        self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[tile]
                    )
            if not self.wallLast and not self.myWallLast:
                for packType, tile, offer in self.packs[0]:
                    if packType == "PENG" and tile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[tile]
                        )
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == "BuHua":
            assert not self.duplicate
            if p == 0:
                self.flower += 1
            self.tileWall -= 1
            self.myWallLast = self.wallLast = self.tileWall == 0
            self.isAboutKong = False
        if t[2] == "Draw":
            if self.duplicate:
                self.tileWall[p] -= 1
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
            else:
                self.tileWall -= 1
                self.myWallLast = self.wallLast = self.tileWall == 0
            return
        if t[2] == "Invalid":
            self.valid = []
            return self._obs()
        if t[2] == "Hu":
            self.valid = []
            return self._obs()
        if t[2] == "Play":
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self._unshown_embedding_update()
            self.history[p].append(
                self.OFFSET_ACT["Play"] + self.OFFSET_TILE[self.curTile]
            )
            self._history_embedding_append(p)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[self.curTile]
                        )
                        if self.hand.count(self.curTile) == 3 and not self.myWallLast:
                            self.valid.append(
                                self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[self.curTile]
                            )
                    color = self.curTile[0]
                    if p == 3 and color in "WTB":
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 3) * 3
                                + 2
                            )
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 2) * 3
                                + 1
                            )
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(
                                self.OFFSET_ACT["Chi"]
                                + "WTB".index(color) * 21
                                + (num - 1) * 3
                            )
                self.valid.append(self.OFFSET_ACT["Pass"])
                return self._obs()
        if t[2] == "Chi":
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(("CHI", tile, int(self.curTile[1]) - num + 2))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self._unshown_embedding_update()
            self.history[p].append(
                self.OFFSET_ACT["Chi"]
                + "WTB".index(color) * 7 * 3
                + (num - 2) * 3
                + int(self.curTile[1])
                - num
                + 1
            )
            self._history_embedding_append(p)
            if self.duplicate:
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == "UnChi":
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self._pack_embedding_pop(p)
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            self._unshown_embedding_update()
            self.history[p].pop()
            self._history_embedding_pop(p)
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Peng":
            self.packs[p].append(("PENG", self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] += 2
            self._unshown_embedding_update()
            self.history[p].append(
                self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[self.curTile]
            )
            self._history_embedding_append(p)
            if self.duplicate:
                self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT["Play"] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == "UnPeng":
            self._pack_embedding_pop(p)
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            self._unshown_embedding_update()
            self.history[p].pop()
            self._history_embedding_pop(p)
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Gang":
            self.packs[p].append(("GANG", self.curTile, (4 + p - self.tileFrom) % 4))
            self._pack_embedding_append(p)
            self.shownTiles[self.curTile] += 3
            self._unshown_embedding_update()
            self.history[p].append(
                self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[self.curTile]
            )
            self._history_embedding_append(p)
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == "AnGang":
            tile = "CONCEALED" if p else t[3]
            self.packs[p].append(("GANG", tile, 0))
            self._pack_embedding_append(p)
            self.history[p].append(self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[tile])
            self._history_embedding_append(p)
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == "BuGang":
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ("GANG", tile, self.packs[p][i][2])
                    offset = (
                        self.OFFSET_OBS["PLAYER_START"]
                        + self.OFFSET_OBS["PLAYER_LEN"] * p
                        + self.OFFSET_OBS["MELD_START"]
                        + self.OFFSET_OBS["MELD_LEN"] * i
                    )
                    self.obs[offset + 3] = self.OFFSET_TILE[tile]
                    break
            self.shownTiles[tile] += 1
            self._unshown_embedding_update()
            self.history[p].append(self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[tile])
            self._history_embedding_append(p)
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                self.valid.append(self.OFFSET_ACT["Pass"])
                return self._obs()
        raise NotImplementedError("Unknown request %s!" % request)

    """
    Pass
    Hu
    Play XX
    Chi XX
    Peng
    Gang
    (An)Gang XX
    BuGang XX
    """

    def action2response(self, action):
        if action < self.OFFSET_ACT["Hu"]:
            return "Pass"
        if action < self.OFFSET_ACT["Play"]:
            return "Hu"
        if action < self.OFFSET_ACT["Chi"]:
            return "Play " + self.TILE_LIST[action - self.OFFSET_ACT["Play"]]
        if action < self.OFFSET_ACT["Peng"]:
            t = (action - self.OFFSET_ACT["Chi"]) // 3
            return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT["Gang"]:
            return "Peng"
        if action < self.OFFSET_ACT["AnGang"]:
            return "Gang"
        if action < self.OFFSET_ACT["BuGang"]:
            return "Gang " + self.TILE_LIST[action - self.OFFSET_ACT["AnGang"]]
        return "BuGang " + self.TILE_LIST[action - self.OFFSET_ACT["BuGang"]]

    """
    Pass
    Hu
    Play XX
    Chi XX XX
    Peng XX
    Gang XX
    (An)Gang XX
    BuGang XX
    """

    def response2action(self, response):
        t = response.split()
        if t[0] == "Pass":
            return self.OFFSET_ACT["Pass"]
        if t[0] == "Hu":
            return self.OFFSET_ACT["Hu"]
        if t[0] == "Play":
            return self.OFFSET_ACT["Play"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Chi":
            return (
                self.OFFSET_ACT["Chi"]
                + "WTB".index(t[1][0]) * 7 * 3
                + (int(t[2][1]) - 2) * 3
                + int(t[1][1])
                - int(t[2][1])
                + 1
            )
        if t[0] == "Peng":
            return self.OFFSET_ACT["Peng"] + self.OFFSET_TILE[t[1]]
        if t[0] == "Gang":
            return self.OFFSET_ACT["Gang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "BuGang":
            return self.OFFSET_ACT["BuGang"] + self.OFFSET_TILE[t[1]]
        if t[0] == "AnGang":
            return self.OFFSET_ACT["AnGang"] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT["Pass"]

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS["HAND"] : self.OFFSET_OBS["WALL"]] = -1
        # print(len(self.hand), self.hand)
        for i, tile in enumerate(self.hand):
            self.obs[self.OFFSET_OBS["HAND"] + i] = self.OFFSET_TILE[tile]

    def _wall_embedding_update(self):
        self.obs[self.OFFSET_OBS["WALL"] : self.OFFSET_OBS["PLAYER_START"]] = -1
        for i, tile in enumerate(
            self.wall[: self.OFFSET_OBS["PLAYER_START"] - self.OFFSET_OBS["WALL"]]
        ):
            self.obs[self.OFFSET_OBS["WALL"] + i] = self.OFFSET_TILE[tile]

    def _pack_embedding_append(self, p):
        l = len(self.packs[p]) - 1
        packType, tile, offer = self.packs[p][-1]
        offset = (
            self.OFFSET_OBS["PLAYER_START"]
            + self.OFFSET_OBS["PLAYER_LEN"] * p
            + self.OFFSET_OBS["MELD_START"]
            + self.OFFSET_OBS["MELD_LEN"] * l
        )
        if packType == "CHI":
            for i in range(-1, 2):
                self.obs[offset + i + 1] = self.OFFSET_TILE[tile] + i
        elif packType == "PENG":
            self.obs[offset : offset + 3] = self.OFFSET_TILE[tile]
        else:
            self.obs[offset : offset + 4] = self.OFFSET_TILE[tile]

    def _pack_embedding_pop(self, p):
        l = len(self.packs[p])
        offset = (
            self.OFFSET_OBS["PLAYER_START"]
            + self.OFFSET_OBS["PLAYER_LEN"] * p
            + self.OFFSET_OBS["MELD_START"]
            + self.OFFSET_OBS["MELD_LEN"] * l
        )
        self.obs[offset : offset + 4] = -1

    def _history_embedding_append(self, p):
        assert len(self.history[p]) <= 29
        l = len(self.history[p]) - 1
        action = self.history[p][-1]
        offset = self.OFFSET_OBS["PLAYER_START"] + self.OFFSET_OBS["PLAYER_LEN"] * p + l
        self.obs[offset] = action

    def _history_embedding_pop(self, p):
        l = len(self.history[p])
        offset = self.OFFSET_OBS["PLAYER_START"] + self.OFFSET_OBS["PLAYER_LEN"] * p + l
        self.obs[offset] = -1

    def _unshown_embedding_update(self):
        for i, tile in enumerate(self.TILE_LIST):
            self.obs[self.OFFSET_OBS["UNSHOWN"] + i] = 4 - self.shownTiles[tile]

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=self.flower,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception("Not Enough Fans")
        except:
            return False
        return True

    # valid actions
    def action_mask(self):
        mask = np.zeros(self.action_space, np.uint8)
        if 1 in self.valid:
            mask[1] = 1
            return mask
        for a in self.valid:
            mask[a] = 1
        return mask

    def _obs(self):
        return {
            "observation": self.obs_normal(),
            "action_mask": self.action_mask(),
        }

    # normal_observation
    def obs_normal(self):
        return self.obs.copy()

    # oracle_observation
    def obs_oracle(self, obs_list):
        obs = np.zeros(self.oracle_obs_space, np.uint8)
        obs[:36] = obs_list[self.seatWind][:36]
        for i in range(4):
            obs[36 + i * 69 : 81 + i * 69] = obs_list[(i + self.seatWind) % 4][60:105]
            obs[81 + i * 69 : 105 + i * 69] = obs_list[(i + self.seatWind) % 4][36:60]
        return obs

    @staticmethod
    def feature_normal_from_normal(normal_obs):
        raise NotImplementedError

    @staticmethod
    def feature_normal_from_oracle(oracle_obs):
        raise NotImplementedError

    @staticmethod
    def feature_oracle_from_oracle(oracle_obs):
        raise NotImplementedError


TILE_LIST = [
    *("W%d" % (i + 1) for i in range(9)),
    *("T%d" % (i + 1) for i in range(9)),
    *("B%d" % (i + 1) for i in range(9)),
    *("F%d" % (i + 1) for i in range(4)),
    *("J%d" % (i + 1) for i in range(3)),
]


class FeatureFull(FeatureAgent2Adapted):

    # quan1+men1+unseen1+hand1+ meld4*4 +(history29)*4
    normal_feature_space = (136, 4, 9)
    # quan1+men1+unseen1+(history29+meld4+hand1+wall10)*4
    oracle_feature_space = (179, 4, 9)

    @staticmethod
    def feature_normal_from_normal(normal_obs, tile_coding):
        OFFSET_TILE = {c: i for i, c in enumerate(tile_coding)}
        OFFSET_TILE["PUBLIC"] = 34
        OFFSET_TILE["CONCEALED"] = 35
        feature = np.zeros((FeatureFull.normal_feature_space[0], 36), np.uint8)
        # 8-bit coding for action, 29 actions, 4 players
        # 2-bit action: 00：play, 01: chi, 02: peng, 03: gang
        dense_feature = np.zeros((8 * 29 * 4), np.uint8)
        feature[0, normal_obs[0]] = 1
        feature[1, normal_obs[1]] = 1
        for j in range(34):
            feature[2, j] = normal_obs[j + 2]
        tile_list = []
        for j in range(14):
            tile = normal_obs[36 + j]
            if tile != 255:
                tile_list.append(TILE_LIST[tile])
                feature[3, tile] += 1
        # print(tile_list)
        for i in range(4):
            for j in range(4):
                offset = 89 + i * 45 + j * 4
                for k in range(4):
                    tile = normal_obs[offset + k]
                    if tile != 255:
                        feature[4 + i * 4 + j, tile] += 1
        for i in range(4):
            for j in range(29):
                action = normal_obs[60 + i * 45 + j]
                if action < FeatureAgent2Adapted.OFFSET_ACT["Chi"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Play"]
                    feature[20 + i * 29 + j, tile] = 1
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [0, 0]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(tile, 6)
                    )
                elif action < FeatureAgent2Adapted.OFFSET_ACT["Peng"]:
                    t = (action - FeatureAgent2Adapted.OFFSET_ACT["Chi"]) // 3
                    color = "WTB"[t // 7]
                    num = t % 7 + 2
                    tile = "%s%d" % (color, num)
                    for k in range(-1, 2):
                        feature[20 + i * 29 + j, OFFSET_TILE[tile] + k] = 1
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [0, 1]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(OFFSET_TILE[tile], 6)
                    )

                elif action < FeatureAgent2Adapted.OFFSET_ACT["Gang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Peng"]
                    feature[20 + i * 29 + j, tile] = 3
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [1, 0]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(tile, 6)
                    )
                elif action < FeatureAgent2Adapted.OFFSET_ACT["AnGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Gang"]
                    feature[20 + i * 29 + j, tile] = 4
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [1, 1]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(tile, 6)
                    )
                elif action < FeatureAgent2Adapted.OFFSET_ACT["BuGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["AnGang"]
                    feature[20 + i * 29 + j, tile] = 4
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [1, 1]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(tile, 6)
                    )
                elif action != 255:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["BuGang"]
                    feature[20 + i * 29 + j, tile] = 4
                    dense_feature[i * 8 * 29 + j : i * 8 * 29 + j + 2] = [1, 1]
                    dense_feature[i * 8 * 29 + j + 2 : i * 8 * 29 + j + 8] = (
                        convert_to_fixed_length_binary(tile, 6)
                    )
        feature = feature.reshape(np.prod(FeatureFull.normal_feature_space))
        return np.concatenate((feature, dense_feature))

    @staticmethod
    def feature_normal_from_oracle(oracle_obs, tile_coding):
        OFFSET_TILE = {c: i for i, c in enumerate(tile_coding)}
        OFFSET_TILE["PUBLIC"] = 34
        OFFSET_TILE["CONCEALED"] = 35
        feature = np.zeros((FeatureFull.normal_feature_space[0], 36), np.uint8)
        feature[0, oracle_obs[0]] = 1
        feature[1, oracle_obs[1]] = 1
        for j in range(34):
            feature[2, j] = oracle_obs[j + 2]
        for j in range(14):
            tile = oracle_obs[81 + j]
            if tile != 255:
                feature[3, tile] += 1
        for i in range(4):
            for j in range(29):
                action = oracle_obs[36 + i * 69 + j]
                if action < FeatureAgent2Adapted.OFFSET_ACT["Chi"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Play"]
                    feature[4 + i * 33 + j, tile] = 1
                elif action < FeatureAgent2Adapted.OFFSET_ACT["Peng"]:
                    t = (action - FeatureAgent2Adapted.OFFSET_ACT["Chi"]) // 3
                    color = "WTB"[t // 7]
                    num = t % 7 + 2
                    tile = "%s%d" % (color, num)
                    for k in range(-1, 2):
                        feature[4 + i * 33 + j, OFFSET_TILE[tile] + k] = 1
                elif action < FeatureAgent2Adapted.OFFSET_ACT["Gang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Peng"]
                    feature[4 + i * 33 + j, tile] = 3
                elif action < FeatureAgent2Adapted.OFFSET_ACT["AnGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Gang"]
                    feature[4 + i * 33 + j, tile] = 4
                elif action < FeatureAgent2Adapted.OFFSET_ACT["BuGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["AnGang"]
                    feature[4 + i * 33 + j, tile] = 4
                elif action != 255:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["BuGang"]
                    feature[4 + i * 33 + j, tile] = 4
            for j in range(4):
                offset = 65 + i * 69 + j * 4
                for k in range(4):
                    tile = oracle_obs[offset + k]
                    if tile != 255:
                        feature[33 + i * 33 + j, tile] += 1
        return feature.reshape(FeatureFull.normal_feature_space)

    @staticmethod
    def feature_oracle_from_oracle(oracle_obs):
        feature = np.zeros((FeatureFull.oracle_feature_space[0], 36), np.uint8)
        feature[0, oracle_obs[0]] = 1
        feature[1, oracle_obs[1]] = 1
        for j in range(34):
            feature[2, j] = oracle_obs[j + 2]
        for i in range(4):
            for j in range(29):
                action = oracle_obs[36 + i * 69 + j]
                if action < FeatureAgent2Adapted.OFFSET_ACT["Chi"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Play"]
                    feature[3 + i * 44 + j, tile] = 1
                elif action < FeatureAgent2Adapted.OFFSET_ACT["Peng"]:
                    t = (action - FeatureAgent2Adapted.OFFSET_ACT["Chi"]) // 3
                    color = "WTB"[t // 7]
                    num = t % 7 + 2
                    tile = "%s%d" % (color, num)
                    for k in range(-1, 2):
                        feature[
                            3 + i * 44 + j, FeatureAgent2Adapted.OFFSET_TILE[tile] + k
                        ] = 1
                elif action < FeatureAgent2Adapted.OFFSET_ACT["Gang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Peng"]
                    feature[3 + i * 44 + j, tile] = 3
                elif action < FeatureAgent2Adapted.OFFSET_ACT["AnGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["Gang"]
                    feature[3 + i * 44 + j, tile] = 4
                elif action < FeatureAgent2Adapted.OFFSET_ACT["BuGang"]:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["AnGang"]
                    feature[3 + i * 44 + j, tile] = 4
                elif action != 255:
                    tile = action - FeatureAgent2Adapted.OFFSET_ACT["BuGang"]
                    feature[3 + i * 44 + j, tile] = 4
            for j in range(4):
                offset = 65 + i * 69 + j * 4
                for k in range(4):
                    tile = oracle_obs[offset + k]
                    if tile != 255:
                        feature[32 + i * 44 + j, tile] += 1
            for j in range(14):
                tile = oracle_obs[81 + i * 69 + j]
                if tile != 255:
                    feature[36 + i * 44, tile] += 1
            for j in range(10):
                tile = oracle_obs[95 + i * 69 + j]
                if tile != 255:
                    feature[37 + i * 44 + j, tile] = 1
        return feature.reshape(FeatureFull.oracle_feature_space)


class FeatureSlim(FeatureAgent2Adapted):

    # quan1+men1+unseen1+hand1+meld4*4
    normal_feature_space = (20, 4, 9)
    # quan1+men1+unseen1+(meld4+hand1)*4
    oracle_feature_space = (23, 4, 9)

    @staticmethod
    def feature_normal_from_normal(normal_obs):
        feature = np.zeros((FeatureSlim.normal_feature_space[0], 36), np.uint8)
        feature[0, normal_obs[0]] = 1
        feature[1, normal_obs[1]] = 1
        for j in range(34):
            feature[2, j] = normal_obs[j + 2]
        for j in range(14):
            tile = normal_obs[36 + j]
            if tile != 255:
                feature[3, tile] += 1
        for i in range(4):
            for j in range(4):
                offset = 89 + i * 45 + j * 4
                for k in range(4):
                    tile = normal_obs[offset + k]
                    if tile != 255:
                        feature[4 + i * 4 + j, tile] += 1
        return feature.reshape(FeatureSlim.normal_feature_space)

    @staticmethod
    def feature_normal_from_oracle(oracle_obs):
        feature = np.zeros((FeatureSlim.normal_feature_space[0], 36), np.uint8)
        feature[0, oracle_obs[0]] = 1
        feature[1, oracle_obs[1]] = 1
        for j in range(34):
            feature[2, j] = oracle_obs[j + 2]
        for j in range(14):
            tile = oracle_obs[81 + j]
            if tile != 255:
                feature[3, tile] += 1
        for i in range(4):
            for j in range(4):
                offset = 65 + i * 69 + j * 4
                for k in range(4):
                    tile = oracle_obs[offset + k]
                    if tile != 255:
                        feature[4 + i * 4 + j, tile] += 1
        return feature.reshape(FeatureSlim.normal_feature_space)

    @staticmethod
    def feature_oracle_from_oracle(oracle_obs):
        feature = np.zeros((FeatureSlim.oracle_feature_space[0], 36), np.uint8)
        feature[0, oracle_obs[0]] = 1
        feature[1, oracle_obs[1]] = 1
        for j in range(34):
            feature[2, j] = oracle_obs[j + 2]
        for i in range(4):
            for j in range(4):
                offset = 65 + i * 69 + j * 4
                for k in range(4):
                    tile = oracle_obs[offset + k]
                    if tile != 255:
                        feature[3 + i * 5 + j, tile] += 1
            for j in range(14):
                tile = oracle_obs[81 + i * 69 + j]
                if tile != 255:
                    feature[7 + i * 5, tile] += 1
        return feature.reshape(FeatureSlim.oracle_feature_space)


def get_feature_clz(name):
    return {"full": FeatureFull, "slim": FeatureSlim}[name]
