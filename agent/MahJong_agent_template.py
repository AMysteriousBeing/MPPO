class MahjongGBAgent:

    def __init__(self, seatWind):
        pass

    """
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(me) UnPeng
    Player N(me) UnChi XX
    
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
        pass

    """
    Hu
    Play XX
    (An)Gang XX
    BuGang XX
    Gang
    Peng
    Chi XX
    Pass
    """

    def action2response(self, action):
        pass


class MahjongGBAgent2:

    raw_observation_space = None
    normal_observation_space = None
    oracle_observation_space = None
    action_space = None

    def __init__(self, seatWind, duplicate):
        pass

    """
    Wind 0..3
    Wall XX XX ...
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

    def request(self, request):
        pass

    """
    Hu
    Play XX
    Chi XX XX
    Peng XX
    Gang XX
    (An)Gang XX
    BuGang XX
    Pass
    """

    def action2response(self, action):
        pass

    def response2action(self, response):
        pass

    # valid actions
    def action_mask(self):
        pass

    # For inference

    # normal_observation
    def obs_normal(self):
        pass

    @staticmethod
    def feature_normal_from_normal(normal_obs):
        pass

    def feature(self):
        return self.feature_normal_from_normal(self.obs_normal())

    # For training

    # oracle_observation
    def obs_oracle(self, obs_list):
        pass

    @staticmethod
    def feature_normal_from_oracle(oracle_obs):
        pass

    @staticmethod
    def feature_oracle_from_oracle(oracle_obs):
        pass
