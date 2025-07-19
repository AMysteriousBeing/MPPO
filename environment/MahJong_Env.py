from agent import MahjongGBAgent

import random
from collections import defaultdict
import copy

try:
    from MahjongGB import MahjongFanCalculator, MahjongShanten, RegularShanten
except:
    print(
        "MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information."
    )
    raise


class Error(Exception):
    pass


class MahjongGBEnv:
    """
    Reward Engineering: Player Hu: Player +0.5, others -0.2
    Player Shanten Progress + 0.2
    Game Step: self.game_step_penalty
    """

    agent_names = ["player_%d" % i for i in range(1, 5)]

    def __init__(self, config):
        self.agentclz_list = []
        for i in range(4):
            assert (
                "agent_clz{}".format(i) in config
            ), "must specify agent_clz to process features!"
            self.agentclz_list.append(config["agent_clz{}".format(i)])
            assert issubclass(
                self.agentclz_list[-1], MahjongGBAgent
            ), "ageng_clz must be a subclass of MahjongGBAgent!"
        self.duplicate = config.get("duplicate", True)
        self.variety = config.get("variety", -1)
        self.r = random.Random()
        self.normalizeReward = config.get("reward_norm", False)
        self.debug = False
        self.game_step_penalty = -0.0006  # -0.0008  # -0.0006
        self.shanten_reward = 0.07  # 0  # 0.07
        self.fan_name_list = []

    def return_agent_list(self):
        return self.agents

    def get_fan_names(self):
        return self.fan_name_list

    def expose_agent_state(self):
        agent_hands = self.hands
        agent_packs = self.packs
        return agent_hands, agent_packs, self.curTile

    def reset(self, prevalentWind=-1, tileWall="", steps_to_kill=-1, seed=-1):
        # Create agents to process features
        self.steps_to_kill = steps_to_kill
        self.agents = [self.agentclz_list[i](i) for i in range(4)]
        self.reward = None
        self.game_step = 0
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        if seed != -1:
            random.seed(seed)
        # Init prevalent wind
        self.prevalentWind = (
            random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        )
        for agent in self.agents:
            agent.request2obs("Wind %d" % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append("W" + str(i))
                    self.tileWall.append("B" + str(i))
                    self.tileWall.append("T" + str(i))
                for i in range(1, 5):
                    self.tileWall.append("F" + str(i))
                for i in range(1, 4):
                    self.tileWall.append("J" + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = " ".join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34 : (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        # Deal cards
        self._deal()
        return self._obs()

    def reset_SIL_adapted(
        self, prevalentWind=-1, tileWall="", steps_to_kill=-1, seed=-1
    ):
        """
        Same interface as reset, but gives SIL style returns
        """
        # Create agents to process features
        self.steps_to_kill = steps_to_kill
        self.agents = [self.agentclz_list[i](i) for i in range(4)]
        self.reward = None
        self.game_step = 0
        self.done = False
        # Init random seed
        if self.variety > 0:
            random.seed(self.r.randint(0, self.variety - 1))
        if seed != -1:
            random.seed(seed)
        # Init prevalent wind
        self.prevalentWind = (
            random.randint(0, 3) if prevalentWind < 0 else prevalentWind
        )
        for agent in self.agents:
            agent.request2obs("Wind %d" % self.prevalentWind)
        # Prepare tile wall
        if tileWall:
            self.tileWall = tileWall.split()
        else:
            self.tileWall = []
            for j in range(4):
                for i in range(1, 10):
                    self.tileWall.append("W" + str(i))
                    self.tileWall.append("B" + str(i))
                    self.tileWall.append("T" + str(i))
                for i in range(1, 5):
                    self.tileWall.append("F" + str(i))
                for i in range(1, 4):
                    self.tileWall.append("J" + str(i))
            random.shuffle(self.tileWall)
        self.originalTileWall = " ".join(self.tileWall)
        if self.duplicate:
            self.tileWall = [self.tileWall[i * 34 : (i + 1) * 34] for i in range(4)]
        self.shownTiles = defaultdict(int)
        tile_wall_copy = copy.deepcopy(self.tileWall)
        # Deal cards
        self._deal()
        return self._obs(), {
            "prevalentWind": self.prevalentWind,
            "tileWall": tile_wall_copy,
        }

    def step(self, action_dict, early_kill=False):
        try:
            self.game_step += 1
            if self.state == 0:
                # After Chi/Peng, prepare to Play
                response = (
                    self.agents[self.curPlayer]
                    .action2response(action_dict[self.agent_names[self.curPlayer]])
                    .split()
                )
                if response[0] == "Play":
                    self._discard(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
                self.isAboutKong = False
            elif self.state == 1:
                # After Draw, prepare to Hu/Play/Gang/BuGang
                response = (
                    self.agents[self.curPlayer]
                    .action2response(action_dict[self.agent_names[self.curPlayer]])
                    .split()
                )
                if response[0] == "Hu":
                    self.shownTiles[self.curTile] += 1
                    self.fan_name_list = self._checkMahjong(
                        self.curPlayer, isSelfDrawn=True, isAboutKong=self.isAboutKong
                    )
                elif response[0] == "Play":
                    self.hands[self.curPlayer].append(self.curTile)
                    self._discard(self.curPlayer, response[1])
                elif (
                    response[0] == "Gang" and not self.myWallLast and not self.wallLast
                ):
                    self._concealedKong(self.curPlayer, response[1])
                elif (
                    response[0] == "BuGang"
                    and not self.myWallLast
                    and not self.wallLast
                ):
                    self._promoteKong(self.curPlayer, response[1])
                else:
                    raise Error(self.curPlayer)
            elif self.state == 2:
                # After Play, prepare to Chi/Peng/Gang/Hu/Pass

                # check for early termination
                if self.steps_to_kill != -1 and self.game_step > self.steps_to_kill:
                    self.obs = {
                        i: self.agents[i].request2obs("Huang") for i in range(4)
                    }
                    # calculating final shanten dist
                    self.reward = [-0.2 + self.game_step_penalty for i in range(4)]
                    self.done = True
                    return self._obs(), self._reward(), self._done()

                responses = {
                    i: self.agents[i].action2response(action_dict[self.agent_names[i]])
                    for i in range(4)
                    if i != self.curPlayer
                }
                t = {i: responses[i].split() for i in responses}
                # Priority: Hu > Peng/Gang > Chi
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if t[i][0] == "Hu":
                        self.fan_name_list = self._checkMahjong(
                            i, play_id=self.curPlayer
                        )
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if (
                            t[i][0] == "Gang"
                            and self._canDrawTile(i)
                            and not self.wallLast
                        ):
                            self._kong(i, self.curTile)
                            break
                        elif t[i][0] == "Peng" and not self.wallLast:
                            self._pung(i, self.curTile)
                            break
                    else:
                        i = (self.curPlayer + 1) % 4
                        if t[i][0] == "Chi" and not self.wallLast:
                            self._chow(i, t[i][1])
                        else:
                            for j in range(1, 4):
                                i = (self.curPlayer + j) % 4
                                if t[i][0] != "Pass":
                                    raise Error(i)
                            if self.wallLast:
                                # A draw
                                self.obs = {
                                    i: self.agents[i].request2obs("Huang")
                                    for i in range(4)
                                }
                                self.reward = [
                                    -0.2 + self.game_step_penalty for i in range(4)
                                ]
                                self.done = True
                            else:
                                # Next player
                                self.curPlayer = (self.curPlayer + 1) % 4
                                self._draw(self.curPlayer)
            elif self.state == 3:
                # After BuGang, prepare to Hu/Pass
                responses = {
                    i: self.agents[i].action2response(action_dict[self.agent_names[i]])
                    for i in range(4)
                    if i != self.curPlayer
                }
                for j in range(1, 4):
                    i = (self.curPlayer + j) % 4
                    if responses[i] == "Hu":
                        self.fan_name_list = self._checkMahjong(
                            i, isAboutKong=True, play_id=self.curPlayer
                        )
                        break
                else:
                    for j in range(1, 4):
                        i = (self.curPlayer + j) % 4
                        if responses[i] != "Pass":
                            raise Error(i)
                    self._draw(self.curPlayer)
        except Error as e:
            player = e.args[0]
            self.obs = {
                i: self.agents[i].request2obs("Player %d Invalid" % player)
                for i in range(4)
            }

            self.reward = [-0.3 + self.game_step_penalty for i in range(4)]
            self.done = True
        # if self.debug:
        #     print(self._reward())
        return self._obs(), self._reward(), self._done()

    def _obs(self):
        return {self.agent_names[k]: v for k, v in self.obs.items()}

    def _reward(self):
        shantens = {}
        for player in self.obs:
            if len(self.hands[player]) % 3 == 1:
                shantens[player] = min(
                    MahjongShanten(
                        pack=tuple(self.packs[player]), hand=tuple(self.hands[player])
                    ),
                    RegularShanten(hand=tuple(self.hands[player]))[0],
                )
            else:
                shantens[player] = self.shantens[player]
        rewards = {
            self.agent_names[k]: (self.shantens[k] - shantens[k]) * self.shanten_reward
            + self.game_step_penalty
            for k in self.obs
        }
        self.shantens.update(shantens)
        if self.reward:
            return {self.agent_names[k]: self.reward[k] for k in self.obs}
        else:
            return rewards
        # return {self.agent_names[k]: 0 for k in self.obs}

    def _done(self):
        return self.done

    def _drawTile(self, player):
        if self.duplicate:
            return self.tileWall[player].pop()
        return self.tileWall.pop()

    def _canDrawTile(self, player):
        if self.duplicate:
            return bool(self.tileWall[player])
        return bool(self.tileWall)

    def _deal(self):
        self.hands = []
        self.packs = []
        self.shantens = {}
        for i in range(4):
            hand = []
            while len(hand) < 13:
                tile = self._drawTile(i)
                hand.append(tile)
            self.hands.append(hand)
            self.packs.append([])
            self.shantens[i] = min(
                MahjongShanten(pack=tuple(), hand=tuple(hand)),
                RegularShanten(hand=tuple(hand))[0],
            )
            self.agents[i].request2obs(" ".join(["Deal", *hand]))
            if self.debug:
                print(" ".join(["Deal", *hand]))
        self.curPlayer = 0
        self.drawAboutKong = False
        self._draw(self.curPlayer)

    def _draw(self, player):
        tile = self._drawTile(player)
        self.myWallLast = not self._canDrawTile(player)
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.isAboutKong = self.drawAboutKong
        self.drawAboutKong = False
        self.state = 1
        self.curTile = tile
        for i in range(4):
            if i != player:
                self.agents[i].request2obs("Player %d Draw" % player)
        self.obs = {player: self.agents[player].request2obs("Draw %s" % tile)}
        if self.debug:
            print("Draw %s" % tile)

    def _discard(self, player, tile):
        if tile not in self.hands[player]:
            raise Error(player)
        self.hands[player].remove(tile)
        self.shownTiles[tile] += 1
        self.wallLast = not self._canDrawTile((player + 1) % 4)
        self.curTile = tile
        self.state = 2
        self.agents[player].request2obs("Player %d Play %s" % (player, tile))
        self.obs = {
            i: self.agents[i].request2obs("Player %d Play %s" % (player, tile))
            for i in range(4)
            if i != player
        }
        if self.debug:
            print("Player %d Play %s" % (player, tile))

    def _kong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4:
            raise Error(player)
        for i in range(4):
            self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(("GANG", tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for agent in self.agents:
            agent.request2obs("Player %d Gang" % player)
        if self.debug:
            print("Player %d Gang" % player)
        self._draw(player)

    def _pung(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 3:
            raise Error(player)
        for i in range(3):
            self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(("PENG", tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] += 2
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs("Player %d Peng" % player)
        self.obs = {player: self.agents[player].request2obs("Player %d Peng" % player)}
        if self.debug:
            print("Player %d Peng" % player)

    def _chow(self, player, tile):
        self.hands[player].append(self.curTile)
        self.shownTiles[self.curTile] -= 1
        color = tile[0]
        num = int(tile[1])
        for i in range(-1, 2):
            t = color + str(num + i)
            if t not in self.hands[player]:
                raise Error(player)
            self.hands[player].remove(t)
            self.shownTiles[t] += 1
        # offer: 123 for which tile is offered
        self.packs[player].append(("CHI", tile, int(self.curTile[1]) - num + 2))
        self.state = 0
        self.curPlayer = player
        for i in range(4):
            if i != player:
                self.agents[i].request2obs("Player %d Chi %s" % (player, tile))
        self.obs = {
            player: self.agents[player].request2obs("Player %d Chi %s" % (player, tile))
        }
        if self.debug:
            print("Player %d Chi %s" % (player, tile))

    def _concealedKong(self, player, tile):
        self.hands[player].append(self.curTile)
        if self.hands[player].count(tile) < 4:
            raise Error(player)
        for i in range(4):
            self.hands[player].remove(tile)
        # offer: 0 for self, 123 for up/oppo/down
        self.packs[player].append(("GANG", tile, (player + 4 - self.curPlayer) % 4))
        self.shownTiles[tile] = 4
        self.curPlayer = player
        self.drawAboutKong = True
        self.isAboutKong = False
        for i in range(4):
            if i != player:
                self.agents[i].request2obs("Player %d AnGang" % player)
        self.agents[player].request2obs("Player %d AnGang %s" % (player, tile))
        self._draw(player)
        if self.debug:
            print("Player %d AnGang" % player)

    def _promoteKong(self, player, tile):
        self.hands[player].append(self.curTile)
        idx = -1
        for i in range(len(self.packs[player])):
            if self.packs[player][i][0] == "PENG" and self.packs[player][i][1] == tile:
                idx = i
        if idx < 0:
            raise Error(player)
        self.hands[player].remove(tile)
        offer = self.packs[player][idx][2]
        self.packs[player][idx] = ("GANG", tile, offer)
        self.shownTiles[tile] = 4
        self.state = 3
        self.curPlayer = player
        self.curTile = tile
        self.drawAboutKong = True
        self.isAboutKong = False
        self.agents[player].request2obs("Player %d BuGang %s" % (player, tile))
        self.obs = {
            i: self.agents[i].request2obs("Player %d BuGang %s" % (player, tile))
            for i in range(4)
            if i != player
        }
        if self.debug:
            print("Player %d BuGang %s" % (player, tile))

    def _debug_handle(self):
        print(self.game_step)
        print(
            "GAME STEP REWARD: {}, PROGRESSION REWARD: {}".format(
                self.game_step * self.game_step_penalty
            )
        )
        return

    def _checkMahjong(self, player, isSelfDrawn=False, isAboutKong=False, play_id=-1):
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[player]),
                hand=tuple(self.hands[player]),
                winTile=self.curTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[self.curTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=player,
                prevalentWind=self.prevalentWind,
                verbose=True,
            )
            fanCnt = 0
            fan_list = []
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fan_list.append(fanName)
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Error("Not Enough Fans")
            self.obs = {
                i: self.agents[i].request2obs("Player %d Hu" % player) for i in range(4)
            }
            if self.debug:
                print("Player %d Hu" % player)
            self.reward = [-0.2 + self.game_step_penalty for i in range(4)]
            if isSelfDrawn:
                self.reward[player] += 0.2  # 0.3
            elif play_id != -1:
                self.reward[play_id] -= 0.3
            self.reward[player] += 0.6  # 0.8
            self.done = True
            return fan_list
        except Exception as e:
            raise Error(player)
