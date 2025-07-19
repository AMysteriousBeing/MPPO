import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

"""
Tabular model with no bias, no value network
"""


class BlackJackNet5b(nn.Module):
    # tabular

    def __init__(self):
        super(BlackJackNet5b, self).__init__()

        self.p = nn.Sequential(
            nn.Linear(11 * 2 * 10 + 1, 2, bias=False),
        )

        # self.v = nn.Sequential(
        #     nn.Linear(11 * 2 * 10 + 1, 1),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.weight, 0)

    def forward(self, input_data):
        """
        obs: batch*76*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        input_data = input_data.type(torch.float32)
        policy = self.p(input_data)
        # value = self.v(input_data)
        return policy  # , value


class BlackJackGAILDiscriminator(nn.Module):

    def __init__(self):
        super(BlackJackGAILDiscriminator, self).__init__()

        self.public_v = nn.Sequential(
            nn.Linear(221 + 2, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, 64),
            nn.LeakyReLU(True),
            nn.Linear(64, 32),
            nn.LeakyReLU(True),
            nn.Linear(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data, action):
        """
        obs: batch*(12+64)*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        one_hot_actions = torch.nn.functional.one_hot(action, num_classes=2).squeeze()
        obs = torch.concat([input_data, one_hot_actions], dim=1)

        v = self.public_v(obs)
        return v


if __name__ == "__main__":
    import sys
    import _pickle as cPickle

    data = torch.rand(2, 11 * 2 * 10 + 1)
    net = BlackJackNet5()
    output1, out2 = net(data)
    # torch.save(net.state_dict, "./samplesize")
    print("Model Output Shape: ", output1.shape, out2.shape)
    print("Model size: ", sys.getsizeof(cPickle.dumps(net.state_dict())) / 1e6, "M")
