from agent.BlackJack_agent_template import BlackJackAgentTemplate

# from BlackJack_agent_template import BlackJackAgentTemplate
import torch.nn.functional as F
import torch


class BlackJackTabular(BlackJackAgentTemplate):

    def __init__(self):
        pass

    def obs2feature(self, obs):
        # 11-21,    0,1,        1-10
        self_value, usable_ace, dealer_value = obs
        self_value = min(max(self_value, 11), 22)  # Ensure self_value is at least 10
        feature = torch.zeros(11, 2, 10)
        if self_value < 22:
            feature[self_value - 11, usable_ace, dealer_value - 1] = 1
            feature = torch.concat((feature.view(-1), torch.tensor([0])), dim=0)
            return feature.float()
        else:
            feature = torch.concat((feature.view(-1), torch.tensor([1])), dim=0)
            return feature.float()

    def action2response(self, action):
        return action


if __name__ == "__main__":
    agent = BlackJackTabular()
    obs = [22, 0, 10]
    print(agent.obs2feature(obs).shape)
