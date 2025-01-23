from offline_model import ActorModel, CriticModel
import time
import torch
from torch.nn import functional as F
import os
import numpy as np
from utils import *
from buffer import ReplayBuffer
from offline_dataset import qlearning_dataset
from utils.scaler import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from td3bc import TD3BCPolicy
from tqdm import tqdm


def stable_softmax(x, dim):
    shift_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exps = torch.exp(shift_x)
    return exps / torch.sum(exps, dim=dim, keepdim=True)


# Tencent masking method
def legal_soft_max(logits, legal_mask):
    _lsm_const_w, _lsm_const_e = 1e20, 1e-5
    _lsm_const_e = 0.00001

    tmp = logits - _lsm_const_w * (1.0 - legal_mask)
    tmp_max = np.max(tmp, keepdims=True)
    # Not necessary max clip 1
    tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
    # tmp = tf.exp(tmp - tmp_max)* legal_mask + _lsm_const_e
    tmp = (np.exp(tmp) + _lsm_const_e) * legal_mask
    # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
    probs = tmp / np.sum(tmp, keepdims=True)
    return probs


def training(
    config,
):
    # epoch = 650
    # step per epoch = 1000
    batch_size = config["effective_batch_size"]
    # publisher = config["pbl"]
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_save_path = os.path.join(
        "logs",
        config["log_dir"],
        config["ckpt_save_path"],
    )
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    summary = SummaryWriter(os.path.join("logs", config["log_dir"], "records"))

    print("Done initializing rank {}".format(device))
    torch.cuda.empty_cache()
    # offline rl dataset
    dataset = qlearning_dataset(config["path_to_data"])
    # create buffer
    buffer = ReplayBuffer(
        buffer_size=450000,
        obs_shape=(5824,),
        obs_dtype=np.float32,
        action_dim=235,
        action_dtype=np.int64,
        device=device,
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # create model
    actor = ActorModel(device).to(device)
    critic1 = CriticModel(device).to(device)
    critic2 = CriticModel(device).to(device)
    actor.train(True)
    critic1.train(True)
    critic2.train(True)
    iterations = 0  # training iterations
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=config["lr"], weight_decay=1e-3
    )
    critic1_optim = torch.optim.Adam(
        critic1.parameters(), lr=config["lr"], weight_decay=1e-3
    )
    critic2_optim = torch.optim.Adam(
        critic2.parameters(), lr=config["lr"], weight_decay=1e-3
    )

    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    policy = TD3BCPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        gamma=config["gamma"],
        scaler=scaler,
    )

    epoch = config["epoch"]
    step_per_epoch = config["step_per_epoch"]

    last_ckpt_time = None
    iteration_counter = 0
    log_entry_counter = 0
    for e in range(1, epoch + 1):
        policy.train()

        pbar = tqdm(range(step_per_epoch), desc=f"Epoch #{e}/{epoch}")
        for it in pbar:
            iteration_counter += 1
            batch = buffer.sample(batch_size)
            loss = policy.learn(batch)
            pbar.set_postfix(**loss)

            t = time.time()
            if iteration_counter % 50 == 0:
                for k, v in loss.items():
                    summary.add_scalar(f"loss/{k}", v, log_entry_counter)
                    log_entry_counter += 1
            # save checkpoints
            if (
                last_ckpt_time == None
                or t - last_ckpt_time > config["ckpt_save_interval"]
            ):
                path = os.path.join(
                    ckpt_save_path,
                    "model_%d.pt" % iterations,
                )

                torch.save(
                    {
                        "iteration": iterations,
                        "actor_dict": actor.state_dict(),
                        "critic1_dict": critic1.state_dict(),
                        "critic2_dict": critic2.state_dict(),
                    },
                    path,
                )
                last_ckpt_time = t
