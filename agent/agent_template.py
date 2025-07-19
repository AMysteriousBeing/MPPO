class AgentTemplate(object):
    observation_space = None
    action_space = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    def feature_process(self, obs):
        """
        Generate features from observation
        """
        return None

    def act2res(self, act):
        """
        Generate environment-compliant response from agent's action taken
        """
        return None

    def req2obs(self, reqest):
        """
        Convert environment-returned request into agent observations
        """
        return None

    def action_mask(self):
        """
        Generate a mask for valid actions
        """
        pass
