import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

"""
This model is designed for MahJong Feature Agent Adapted
"""


def channel_expansion(compact_data):
    """
    Convert batch*4*9 to batch*4*4*9
    """
    data_list = []
    for i in range(4):
        data_list.append(torch.where(compact_data > (0.25 * i), 1, 0))
    return torch.stack(data_list, dim=1)


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


class GAILDiscriminator(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

        self.public_v = nn.Sequential(
            nn.Conv2d(136, 128, 3, 1, padding=(1, 0)),  # 128*4,7
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, padding=(1, 0)),  # 256*4,5
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, padding=1),  # 128*4,5
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, padding=1),  # 64*4,5
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),  # 64*2,3
            nn.Flatten(start_dim=1),
        )

        self.public_dense_v = nn.Sequential(
            nn.Linear(720 + 928 + 235, 256),
            nn.ReLU(True),
        )

        self.public_dense_v_layers = nn.Sequential(
            nn.Linear(256, 256),
        )

        self.public_concat_v = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(640, 512),
            nn.ReLU(True),
        )
        self._value_branch = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data, actions):
        one_hot_actions = torch.nn.functional.one_hot(
            actions, num_classes=235
        ).squeeze()
        obs = input_data.type(torch.float32)

        obs_dense = torch.concat(
            [obs[:, : 20 * 4 * 9], obs[:, -928:], one_hot_actions], dim=1
        )
        obs = obs[:, : 136 * 4 * 9].reshape(-1, 136, 4, 9)
        cnn_v = self.public_v(obs)
        linear1_v = self.public_dense_v(obs_dense)
        lienar2_v = self.public_dense_v_layers(linear1_v)
        concat1_v = torch.concat([cnn_v, lienar2_v], dim=1)
        concat2_v = self.public_concat_v(concat1_v)
        value = self._value_branch(concat2_v)
        return value

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))


class MahJongCNNNet6_LargeV2(nn.Module):

    def __init__(self, device="cpu"):
        super(MahJongCNNNet6_LargeV2, self).__init__()
        self.device = device
        self.public_p = nn.Sequential(
            nn.Conv2d(136, 256, 3, 1, padding=(1, 0)),  # 128*4,7
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 1, padding=(1, 0)),  # 256*4,5
            nn.ReLU(True),
            nn.Conv2d(512, 256, 3, 1, padding=1),  # 128*4,5
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, padding=1),  # 64*4,5
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1),  # 64*2,3
            nn.Flatten(start_dim=1),
        )

        self.public_dense_p = nn.Sequential(
            nn.Linear(720 + 928, 512),
            nn.ReLU(True),
        )

        self.public_dense_p_layers = PSCN(512, 512, 3)

        self.public_concat_p = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(1280, 1024),
            nn.ReLU(True),
        )
        self._logits_branch = PSCN(1024, 512, 4)
        self._logits_head = nn.Linear(512, 235)

        self.public_v = nn.Sequential(
            nn.Conv2d(136, 128, 3, 1, padding=(1, 0)),  # 128*4,7
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 1, padding=(1, 0)),  # 256*4,5
            nn.ReLU(True),
            nn.Conv2d(256, 128, 3, 1, padding=1),  # 128*4,5
            nn.ReLU(True),
            nn.Conv2d(128, 64, 3, 1, padding=1),  # 64*4,5
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),  # 64*2,3
            nn.Flatten(start_dim=1),
        )

        self.public_dense_v = nn.Sequential(
            nn.Linear(720 + 928, 256),
            nn.ReLU(True),
        )

        self.public_dense_v_layers = PSCN(256, 256, 3)

        self.public_concat_v = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(640, 512),
            nn.ReLU(True),
        )
        self._value_branch = PSCN(512, 256, 3)
        self._value_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.PReLU(),
            nn.Linear(32, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        """
        obs: batch*(12+64)*4*9, first 12 are feature channel, last 64 (32*2) are search channel
        """
        # obs = 136 * 4 * 9 + 928
        obs = input_data["observation"].type(torch.float32)
        mask = input_data["action_mask"].type(torch.float32)
        obs_dense = torch.concat([obs[:, : 20 * 4 * 9], obs[:, -928:]], dim=1)
        obs = obs[:, : 136 * 4 * 9].reshape(-1, 136, 4, 9)

        cnn_p = self.public_p(obs)
        linear1_p = self.public_dense_p(obs_dense)
        linear2_p = self.public_dense_p_layers(linear1_p)
        concat1_p = torch.concat([cnn_p, linear2_p], dim=1)
        concat2_p = self.public_concat_p(concat1_p)
        logit1_p = self._logits_branch(concat2_p)
        logits = self._logits_head(logit1_p)
        inf_mask = torch.clamp(torch.log(mask), -1e20, 1e20)
        masked_logits = logits + inf_mask
        # masked_logits = torch.where(mask, logits, torch.tensor(-1e38).to(self.device))

        cnn_v = self.public_v(obs)
        linear1_v = self.public_dense_v(obs_dense)
        lienar2_v = self.public_dense_v_layers(linear1_v)
        concat1_v = torch.concat([cnn_v, lienar2_v], dim=1)
        concat2_v = self.public_concat_v(concat1_v)
        value1 = self._value_branch(concat2_v)
        value = self._value_head(value1)
        return masked_logits, value


if __name__ == "__main__":
    import sys
    import _pickle as cPickle

    data = torch.rand(2, 136 * 4 * 9 + 928)
    mask = torch.ones(2, 235)
    net = MahJongCNNNet6_LargeV2("cpu")
    output1, out2 = net({"observation": data, "action_mask": mask})
    # torch.save(net.state_dict, "./samplesize")
    print("Model Output Shape: ", output1.shape, out2.shape)
    print("Model size: ", sys.getsizeof(cPickle.dumps(net.state_dict())) / 1e6, "M")
