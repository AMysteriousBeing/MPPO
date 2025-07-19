import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

"""
This model is designed for partial observable maze env (MazeEnv2) with view_size = 5
"""


# 权重初始化
def init_weights(layer, init_type="uniform"):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if init_type == "uniform":
            init_type = np.random.choice(
                a=["kaiming", "xavier", "orthogonal"], p=[0.2, 0.2, 0.6]
            )

        if init_type == "kaiming":
            nn.init.kaiming_uniform_(layer.weight)
        elif init_type == "xavier":
            nn.init.xavier_uniform_(layer.weight)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")

        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    return layer


# 全连接层
class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        activation=nn.PReLU(),
        last_act=False,
        use_norm=False,
        linear=nn.Linear,
        *args,
        **kwargs,
    ):
        super(MLP, self).__init__()
        assert dim_list, "Dim list can't be empty!"
        layers = []
        for i in range(len(dim_list) - 1):
            layer = init_weights(linear(dim_list[i], dim_list[i + 1], *args, **kwargs))
            layers.append(layer)
            if i < len(dim_list) - 2:
                if use_norm:
                    layers.append(nn.LayerNorm(dim_list[i + 1]))
                layers.append(activation)
        if last_act:
            if use_norm:
                layers.append(nn.LayerNorm(dim_list[-1]))
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 一种兼顾宽度和深度的全连接层，提取信息效率更高
class PSCN(nn.Module):
    def __init__(self, input_dim, output_dim, depth, linear=nn.Linear):
        super(PSCN, self).__init__()
        min_dim = 2 ** (depth - 1)
        assert depth >= 1, "depth must be at least 1"
        assert (
            output_dim >= min_dim
        ), f"output_dim must be >= {min_dim} for depth {depth}"
        assert (
            output_dim % min_dim == 0
        ), f"output_dim must be divisible by {min_dim} for depth {depth}"

        self.layers = nn.ModuleList()
        self.output_dim = output_dim
        in_dim, out_dim = input_dim, output_dim

        for _ in range(depth):
            self.layers.append(MLP([in_dim, out_dim], last_act=True, linear=linear))
            in_dim = out_dim // 2
            out_dim //= 2

    def forward(self, x):
        out_parts = []

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                split_size = self.output_dim // (2 ** (i + 1))
                part, x = torch.split(x, [split_size, split_size], dim=-1)
                out_parts.append(part)
            else:
                out_parts.append(x)

        out = torch.cat(out_parts, dim=-1)
        return out


class MazeNet2(nn.Module):

    def __init__(self):
        super(MazeNet2, self).__init__()
        self.public_p = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, padding=(0, 0)),  # 64*15,15
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=(0, 0)),  # 64*13,13
            nn.PReLU(),
            nn.Conv2d(64, 32, 3, 1, padding=0),  # 32*5,5
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, 1, padding=0),  # 16*3,3
            nn.PReLU(),
            nn.Flatten(start_dim=1),
        )

        self.public_dense_p = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
        )

        self.public_concat_p = nn.Sequential(
            nn.PReLU(),
            nn.Linear(64 + 16 * 3 * 3, 512),
            nn.PReLU(),
        )
        self._logits_branch = PSCN(512, 128, 4)
        self._logits_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 4),
        )

        self.public_v = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, padding=(0, 0)),  # 64*15,7
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=(0, 0)),  # 64*7,5
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, 1, padding=0),  # 32*5,5
            nn.PReLU(),
            nn.Conv2d(16, 8, 3, 1, padding=0),  # 16*3,3
            nn.PReLU(),
            nn.Flatten(start_dim=1),
        )

        self.public_dense_v = nn.Sequential(
            nn.Linear(11, 32),
            nn.PReLU(),
        )

        self.public_concat_v = nn.Sequential(
            nn.Linear(32 + 8 * 3 * 3, 256),
            nn.PReLU(),
        )
        self._value_branch = PSCN(256, 64, 3)
        self._value_head = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        """
        obs: batch*(12+64)*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        obs_cnn = input_data["cnn_obs"].type(torch.float32)
        obs_dense = input_data["dense_obs"].type(torch.float32)
        obs_dense_policy = obs_dense[:, :-1]
        mask = input_data["action_mask"].type(torch.float32)

        cnn_p = self.public_p(obs_cnn)
        linear1_p = self.public_dense_p(obs_dense_policy)
        concat1_p = torch.concat([cnn_p, linear1_p], dim=1)
        concat2_p = self.public_concat_p(concat1_p)
        logit1_p = self._logits_branch(concat2_p)
        logits = self._logits_head(logit1_p)
        inf_mask = torch.clamp(torch.log(mask), -1e20, 1e20)
        masked_logits = logits + inf_mask
        # masked_logits = torch.where(mask, logits, torch.tensor(-1e38).to(self.device))

        cnn_v = self.public_v(obs_cnn)
        linear1_v = self.public_dense_v(obs_dense)
        concat1_v = torch.concat([cnn_v, linear1_v], dim=1)
        concat2_v = self.public_concat_v(concat1_v)
        value1 = self._value_branch(concat2_v)
        value = self._value_head(value1)
        return masked_logits, value


class MazeGAILDiscriminator(nn.Module):

    def __init__(self):
        super(MazeGAILDiscriminator, self).__init__()

        self.public_v = nn.Sequential(
            nn.Conv2d(5, 32, 3, 1, padding=(0, 0)),  # 64*15,7
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, 1, padding=(0, 0)),  # 64*7,5
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, 1, padding=0),  # 32*5,5
            nn.PReLU(),
            nn.Conv2d(16, 8, 3, 1, padding=0),  # 16*3,3
            nn.PReLU(),
            nn.Flatten(start_dim=1),
        )

        self.public_dense_v = nn.Sequential(
            nn.Linear(15, 32),
            nn.PReLU(),
        )

        self.public_concat_v = nn.Sequential(
            nn.Linear(32 + 8 * 3 * 3, 256),
            nn.PReLU(),
        )
        self._value_branch = PSCN(256, 64, 3)
        self._value_head = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data, action):
        """
        obs: batch*(12+64)*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        one_hot_actions = torch.nn.functional.one_hot(action, num_classes=4).squeeze()
        obs_cnn = (
            input_data[:, : 5 * 11 * 11].reshape(-1, 5, 11, 11).type(torch.float32)
        )
        obs_dense = torch.concat([input_data[:, 5 * 11 * 11 :], one_hot_actions], dim=1)

        cnn_v = self.public_v(obs_cnn)
        linear1_v = self.public_dense_v(obs_dense)
        concat1_v = torch.concat([cnn_v, linear1_v], dim=1)
        concat2_v = self.public_concat_v(concat1_v)
        value1 = self._value_branch(concat2_v)
        value = self._value_head(value1)
        return value


class MazeNet2DQN(nn.Module):

    def __init__(self, device="cpu"):
        super(MazeNet2DQN, self).__init__()
        self.public_p = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, padding=(0, 0)),  # 64*15,15
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=(0, 0)),  # 64*13,13
            nn.PReLU(),
            nn.Conv2d(64, 32, 3, 1, padding=0),  # 32*5,5
            nn.PReLU(),
            nn.Conv2d(32, 16, 3, 1, padding=0),  # 16*3,3
            nn.PReLU(),
            nn.Flatten(start_dim=1),
        )

        self.public_dense_p = nn.Sequential(
            nn.Linear(10, 64),
            nn.PReLU(),
        )

        self.public_concat_p = nn.Sequential(
            nn.PReLU(),
            nn.Linear(64 + 16 * 3 * 3, 512),
            nn.PReLU(),
        )
        self._logits_branch = PSCN(512, 128, 4)
        self._logits_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.PReLU(),
            nn.Linear(32, 4),
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        """
        obs: batch*(12+64)*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        obs = input_data["observation"].type(torch.float32)
        mask = input_data["action_mask"].type(torch.float32)
        obs_cnn = obs[:, : 5 * 11 * 11].reshape(-1, 5, 11, 11).type(torch.float32)
        obs_dense = obs[:, 5 * 11 * 11 :].type(torch.float32)
        obs_dense_policy = obs_dense[:, :-1]

        cnn_p = self.public_p(obs_cnn)
        linear1_p = self.public_dense_p(obs_dense_policy)
        concat1_p = torch.concat([cnn_p, linear1_p], dim=1)
        concat2_p = self.public_concat_p(concat1_p)
        logit1_p = self._logits_branch(concat2_p)
        logits = self._logits_head(logit1_p)
        inf_mask = torch.clamp(torch.log(mask), -1e20, 1e20)
        masked_logits = logits + inf_mask
        # masked_logits = torch.where(mask, logits, torch.tensor(-1e38).to(self.device))
        return masked_logits


if __name__ == "__main__":
    import sys
    import _pickle as cPickle

    cnn_feature = torch.rand(2, 5, 11, 11).float()
    dense_feature = torch.rand(2, 11).float()
    mask = torch.ones(2, 4)
    net = MazeNet2()
    output1, out2 = net(
        {"cnn_obs": cnn_feature, "dense_obs": dense_feature, "action_mask": mask}
    )
    # torch.save(net.state_dict, "./samplesize")
    print("Model Output Shape: ", output1.shape, out2.shape)
    print("Model size: ", sys.getsizeof(cPickle.dumps(net.state_dict())) / 1e6, "M")
