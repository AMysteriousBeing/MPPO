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


def _tile_offset_correction(no):
    """
    For 5*9 instead of 4*9
    5*9 representation to 34
    """
    if no > 30:
        no -= 5
    return no


def _tile_offset_reverse(no):
    """
    For 5*9 instead of 4*9
    34 representation to 5*9
    """
    if no > 30:
        no += 5
    return no


class FeatureAgent(MahjongGBAgent):

    """
    observation: 10*5*9
        (men + quan + hand4 + global known tile4 + opponent4*3)*5*9
    action_mask: 235
        pass1+hu1+discard34+chi63(3*7*3)+peng34+gang34+angang34+bugang34
    """

    OBS_SIZE = 22
    ACT_SIZE = 235

    OFFSET_OBS = {
        "SEAT_WIND": 0,
        "PREVALENT_WIND": 1,
        "HAND": 2,
        "GLOBAL": 6,
        "OPP1": 10,
        "OPP2": 14,
        "OPP3": 18,
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
    #     *('W%d'%(i+1) for i in range(9)),
    #     *('T%d'%(i+1) for i in range(9)),
    #     *('B%d'%(i+1) for i in range(9)),
    #     *('F%d'%(i+1) for i in range(4)),
    #     *('D%d'%(i+1) for i in range(5)), #D as dummy
    #     *('J%d'%(i+1) for i in range(3))
    # ]

    def __init__(self, seatWind, TILE_LIST):
        self.seatWind = seatWind
        self.TILE_LIST = TILE_LIST
        self.lastPlay = None
        self.packs = [[] for i in range(4)]
        self.history = [[] for i in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.knownTiles = defaultdict(float)  # record all visible tiles
        self.wallLast = False
        self.isAboutKong = False
        self.obs = np.zeros((self.OBS_SIZE, 5 * 9))
        self.OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}
        self.obs[self.OFFSET_OBS["SEAT_WIND"]][
            self.OFFSET_TILE["F%d" % (self.seatWind + 1)]
        ] = 1

    def _global_known_tiles_update(self, tile_list):
        """
        update knownTiles from tile_list
        only call during deal, self draw, other play, other Chi/Peng/Gang
        tile_list: [tile]
        known_tiles: default 0.0, each tile appearance add 0.25 (each tile at most 4 max)
        """
        for tile in tile_list:
            for i in range(4):
                if self.obs[self.OFFSET_OBS["GLOBAL"] + i, self.OFFSET_TILE[tile]] != 0:
                    continue
                else:
                    self.obs[self.OFFSET_OBS["GLOBAL"] + i, self.OFFSET_TILE[tile]] = 1
                    break

            # self.obs[self.OFFSET_OBS['GLOBAL'], self.OFFSET_TILE[tile]]+=0.25
            # if self.obs[self.OFFSET_OBS['GLOBAL'], self.OFFSET_TILE[tile]]>1:
            #     self.obs[self.OFFSET_OBS['GLOBAL'], self.OFFSET_TILE[tile]]=1

    def _oppo_discard_history(self, oppo_no, tile):
        """
        update opponent discard list
        """
        for i in range(4):
            if (
                self.obs[self.OFFSET_OBS["OPP%d" % oppo_no] + i, self.OFFSET_TILE[tile]]
                != 0
            ):
                continue
            else:
                self.obs[
                    self.OFFSET_OBS["OPP%d" % oppo_no] + i, self.OFFSET_TILE[tile]
                ] = 1
                break
        # self.obs[self.OFFSET_OBS['OPP%d'%oppo_no], self.OFFSET_TILE[tile]]+=0.25
        # if self.obs[self.OFFSET_OBS['OPP%d'%oppo_no], self.OFFSET_TILE[tile]] > 1:
        #     self.obs[self.OFFSET_OBS['OPP%d'%oppo_no], self.OFFSET_TILE[tile]] = 1

    """
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
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
            self.obs[self.OFFSET_OBS["PREVALENT_WIND"]][
                self.OFFSET_TILE["F%d" % (self.prevalentWind + 1)]
            ] = 1
            return
        if t[0] == "Deal":
            self.hand = t[1:]
            self._hand_embedding_update()
            self._global_known_tiles_update(t[1:])  # modify global view
            return
        if t[0] == "Huang":
            self.valid = []
            return self._obs()
        if t[0] == "Draw":
            # Available: Hu, Play, AnGang, BuGang
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self._global_known_tiles_update([tile])  # Modify global view
            self.valid = []
            if self._check_mahjong(
                tile, isSelfDrawn=True, isAboutKong=self.isAboutKong
            ):
                self.valid.append(self.OFFSET_ACT["Hu"])
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            for tile in set(self.hand):
                self.valid.append(
                    self.OFFSET_ACT["Play"]
                    + _tile_offset_correction(self.OFFSET_TILE[tile])
                )
                if (
                    self.hand.count(tile) == 4
                    and not self.wallLast
                    and self.tileWall[0] > 0
                ):
                    self.valid.append(
                        self.OFFSET_ACT["AnGang"]
                        + _tile_offset_correction(self.OFFSET_TILE[tile])
                    )
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == "PENG" and tile in self.hand:
                        self.valid.append(
                            self.OFFSET_ACT["BuGang"]
                            + _tile_offset_correction(self.OFFSET_TILE[tile])
                        )
            return self._obs()
        # Player N Invalid/Hu/Draw/Play/Chi/Peng/Gang/AnGang/BuGang XX
        p = (int(t[1]) + 4 - self.seatWind) % 4
        if t[2] == "Draw":
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
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
            self.lastPlay = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            if p == 0:
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # Available: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                self._global_known_tiles_update([self.curTile])  # Modify global view
                self._oppo_discard_history(p, self.curTile)  # Update opponent stat
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT["Hu"])
                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(
                            self.OFFSET_ACT["Peng"]
                            + _tile_offset_correction(self.OFFSET_TILE[self.curTile])
                        )
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(
                                self.OFFSET_ACT["Gang"]
                                + _tile_offset_correction(
                                    self.OFFSET_TILE[self.curTile]
                                )
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
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(
                        self.OFFSET_ACT["Play"]
                        + _tile_offset_correction(self.OFFSET_TILE[tile])
                    )
                return self._obs()
            else:
                tile = t[3]
                color = tile[0]
                num = int(tile[1])
                streak = [color + str(num - 1), color + str(num), color + str(num + 1)]
                if self.lastPlay in streak:
                    streak.remove(self.lastPlay)
                self._global_known_tiles_update(streak)  # Update global view
                return
        if t[2] == "UnChi":
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Peng":
            self.packs[p].append(("PENG", self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(
                        self.OFFSET_ACT["Play"]
                        + _tile_offset_correction(self.OFFSET_TILE[tile])
                    )
                return self._obs()
            else:
                self._global_known_tiles_update([t[3], t[3]])  # Update global view
                return
        if t[2] == "UnPeng":
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == "Gang":
            self.packs[p].append(("GANG", self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            else:
                self._global_known_tiles_update(
                    [t[3], t[3], t[3]]
                )  # Update global view
            return
        if t[2] == "AnGang":
            tile = "CONCEALED" if p else t[3]
            self.packs[p].append(("GANG", tile, 0))
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
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self._global_known_tiles_update([t[3]])  # Update global view
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
            return (
                "Play "
                + self.TILE_LIST[_tile_offset_reverse(action - self.OFFSET_ACT["Play"])]
            )
        if action < self.OFFSET_ACT["Peng"]:
            t = (action - self.OFFSET_ACT["Chi"]) // 3
            return "Chi " + "WTB"[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT["Gang"]:
            return "Peng"
        if action < self.OFFSET_ACT["AnGang"]:
            return "Gang"
        if action < self.OFFSET_ACT["BuGang"]:
            return (
                "Gang "
                + self.TILE_LIST[
                    _tile_offset_reverse(action - self.OFFSET_ACT["AnGang"])
                ]
            )
        return (
            "BuGang "
            + self.TILE_LIST[_tile_offset_reverse(action - self.OFFSET_ACT["BuGang"])]
        )

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

    def response2action(self, response):
        t = response.split()
        if t[0] == "Pass":
            return self.OFFSET_ACT["Pass"]
        if t[0] == "Hu":
            return self.OFFSET_ACT["Hu"]
        if t[0] == "Play":
            return self.OFFSET_ACT["Play"] + _tile_offset_correction(
                self.OFFSET_TILE[t[1]]
            )
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
            return self.OFFSET_ACT["Peng"] + _tile_offset_correction(
                self.OFFSET_TILE[t[1]]
            )
        if t[0] == "Gang":
            return self.OFFSET_ACT["Gang"] + _tile_offset_correction(
                self.OFFSET_TILE[t[1]]
            )
        if t[0] == "AnGang":
            return self.OFFSET_ACT["AnGang"] + _tile_offset_correction(
                self.OFFSET_TILE[t[1]]
            )
        if t[0] == "BuGang":
            return self.OFFSET_ACT["BuGang"] + _tile_offset_correction(
                self.OFFSET_TILE[t[1]]
            )
        return self.OFFSET_ACT["Pass"]

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        return {
            "observation": self.obs.reshape((self.OBS_SIZE, 5, 9)).copy(),
            "action_mask": mask,
        }

    def _hand_embedding_update(self):
        self.obs[self.OFFSET_OBS["HAND"] : self.OFFSET_OBS["HAND"] + 4] = 0
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[
                self.OFFSET_OBS["HAND"] : self.OFFSET_OBS["HAND"] + d[tile],
                self.OFFSET_TILE[tile],
            ] = 1

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=self.shownTiles[winTile] == 4,
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
