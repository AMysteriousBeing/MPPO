from offline_model_discrete import ActorModel, CriticModel
import time
import torch
from torch.nn import functional as F
import os
import numpy as np
from utils import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from buffer import ReplayBuffer
from offline_dataset import qlearning_dataset
from utils.scaler import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from td3bc import TD3BCPolicy
from cql_discrete import CQLDiscretePolicy


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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


def ddp_training(
    rank,
    config,
):
    # epoch = 650
    # step per epoch = 1000
    world_size = config["world_size"]
    setup(rank, world_size)
    batch_size = config["effective_batch_size"] // config["world_size"]
    # publisher = config["pbl"]
    torch.autograd.set_detect_anomaly(True)
    # seed
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True

    if rank == 0:
        ckpt_save_path = os.path.join(
            "logs",
            config["log_dir"],
            config["ckpt_save_path"],
        )
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        summary = SummaryWriter(os.path.join("logs", config["log_dir"], "records"))

    print("Done initializing rank {}".format(rank))
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # offline rl dataset
    dataset = qlearning_dataset(config["path_to_data"], world_size, rank)
    # create buffer
    buffer = ReplayBuffer(
        buffer_size=450000,
        obs_shape=(5824,),
        obs_dtype=np.float32,
        action_dim=1,
        action_dtype=np.int64,
        device=rank,
    )
    buffer.load_dataset(dataset)
    obs_mean, obs_std = buffer.normalize_obs()

    # create model
    actor = ActorModel(rank)
    actor.load_state_dict(
        torch.load("../Supervised_Learning/sl_ckpts/smth/46.pkl", map_location="cpu"),
        strict=False,
    )
    actor.to(rank)
    critic1 = CriticModel(rank).to(rank)
    critic2 = CriticModel(rank).to(rank)
    actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(actor).to(rank)
    critic1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(critic1).to(rank)
    critic2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(critic2).to(rank)
    actor = DDP(actor, device_ids=[rank])
    critic1 = DDP(critic1, device_ids=[rank])
    critic2 = DDP(critic2, device_ids=[rank])
    actor.train(True)
    critic1.train(True)
    critic2.train(True)
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

    policy = CQLDiscretePolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        235,
        config["lr"],
        rank,
        gamma=config["gamma"],
    )

    epoch = config["epoch"]
    step_per_epoch = config["step_per_epoch"]

    last_ckpt_time = None
    iteration_counter = 0
    log_entry_counter = 0
    for e in range(1, epoch + 1):
        policy.train()

        for it in range(step_per_epoch):

            batch = buffer.sample(batch_size)
            loss = policy.learn(batch)

            if rank == 0:
                if iteration_counter % step_per_epoch == 0:
                    print(f"Starting epoch {e}")
                t = time.time()
                if iteration_counter % 50 == 0:
                    for k, v in loss.items():
                        summary.add_scalar(f"loss/{k}", v, log_entry_counter)
                        log_entry_counter += 1
                    summary.add_scalar(
                        f"progress/GPU_iteration", iteration_counter, log_entry_counter
                    )
                # save checkpoints
                if (
                    last_ckpt_time == None
                    or t - last_ckpt_time > config["ckpt_save_interval"]
                ):
                    path = os.path.join(
                        ckpt_save_path,
                        "model_%d.pt" % log_entry_counter,
                    )

                    torch.save(
                        {
                            "iteration": iteration_counter,
                            "actor_dict": actor.state_dict(),
                            "critic1_dict": critic1.state_dict(),
                            "critic2_dict": critic2.state_dict(),
                        },
                        path,
                    )
                    last_ckpt_time = t
            iteration_counter += 1
