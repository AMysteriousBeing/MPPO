from offline_model_discrete import ActorModel, CriticQ, CriticV
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
from bppo import BPPO


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
        action_dim=235,
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
    criticq = CriticQ(rank).to(rank)
    criticq_target = CriticQ(rank).to(rank)
    criticq_target.load_state_dict(criticq.state_dict())
    criticv = CriticV(rank).to(rank)
    actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(actor).to(rank)
    criticq = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criticq).to(rank)
    criticq_target = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criticq_target).to(
        rank
    )
    criticv = torch.nn.SyncBatchNorm.convert_sync_batchnorm(criticv).to(rank)
    actor = DDP(actor, device_ids=[rank])
    criticq = DDP(criticq, device_ids=[rank])
    criticv = DDP(criticv, device_ids=[rank])
    actor.train(True)
    criticq.train(True)
    criticv.train(True)
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=config["lr"], weight_decay=1e-3
    )
    criticq_optim = torch.optim.Adam(
        criticq.parameters(), lr=config["lr"], weight_decay=1e-3
    )
    criticv_optim = torch.optim.Adam(
        criticv.parameters(), lr=config["lr"], weight_decay=1e-3
    )

    target_update_freq = 2
    tau = 0.05

    # update v network and q network
    for step in range(int(1e5)):
        batch = buffer.sample(batch_size)
        obss, actions, next_obss, next_actions, rewards, terminals, masks = (
            batch["observations"],
            batch["actions"],
            batch["next_observations"],
            batch["next_actions"],
            batch["rewards"],
            batch["terminals"],
            batch["masks"],
        )
        actions = torch.nn.functional.one_hot(actions, num_classes=235)
        next_actions = torch.nn.functional.one_hot(next_actions, num_classes=235)
        value_loss = F.mse_loss(criticv(obss), rewards)
        criticv_optim.zero_grad()
        value_loss.backward()
        criticv_optim.step()
        value_loss = value_loss.item()
        with torch.no_grad():
            target_Q_value = rewards + (1 - terminals) * config[
                "gamma"
            ] * criticq_target(next_obss, next_actions)

        Q_val = criticq(obss, actions)
        Q_loss = F.mse_loss(Q_val, target_Q_value)
        criticq_optim.zero_grad()
        Q_loss.backward()
        criticq_optim.step()
        if step % target_update_freq == 0 and rank == 0:
            # criticq_target.load_state_dict(criticq.state_dict())
            for param, target_param in zip(
                criticq.parameters(), criticq_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
        torch.distributed.barrier()

    # scaler for normalizing observations
    scaler = StandardScaler(mu=obs_mean, std=obs_std)

    policy = BPPO(
        actor,
        criticq,
        criticv,
        actor_optim,
        config["lr"],
        config["clip"],
        config["entropy"],
        0.5,
        batch_size,
        rank,
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
                            "criticq_dict": criticq.state_dict(),
                            "criticv_dict": criticv.state_dict(),
                        },
                        path,
                    )
                    last_ckpt_time = t
            iteration_counter += 1
