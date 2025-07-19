from model import BlackJackNet5b
import time
import torch
from torch.nn import functional as F
import os
import statistics
import json
import numpy as np
from multiprocessing import shared_memory
from utils import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

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


def BJ_ddp_training_PPOfD(
    rank,
    mutable_param_path,
    model_pool_name,
    replay_data_queue_list,
    stat_queue,
    config,
    shared_arr,
    sync=False,
):
    world_size = config["world_size"]
    setup(rank, world_size)
    log_path = "logs/{}/learner_log.txt".format(config["log_dir"])
    logger = CustomLogger(log_path)
    replay_data_queue = replay_data_queue_list[rank]
    rb_size = config["replay_buffer_size"] // config["world_size"]
    rb_svr = ReplayBufferLearnerSideTagged(rb_size)
    batch_size = config["effective_batch_size"] // config["world_size"]
    # publisher = config["pbl"]
    torch.autograd.set_detect_anomaly(True)
    entropy_coeff = config["entropy_coeff"]
    model_tag_id = 0
    static_ratio = config.get("static_ratio", 0.05)
    sample_batch_size = batch_size - int(batch_size * static_ratio)
    demo_batch_size = int(batch_size * static_ratio)

    with open(os.path.join(mutable_param_path, "wait_time.json"), "r") as f:
        mutable_data = json.load(f)
        wait_time = mutable_data["wait"]

    logger.info("Setting up learner log entries")

    model_pool_server = None
    if rank == 0:
        ckpt_save_path = os.path.join(
            "logs",
            config["log_dir"],
            config["ckpt_save_path"],
        )
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        # setup model pool server
        model_pool_server = ModelPoolServer(4, model_pool_name)

    first_model_sent = False
    learner_status_list = shared_arr
    print("Done initializing rank {}".format(rank))
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    dataset = qlearning_dataset(config["path_to_data"], world_size, rank)
    # create buffer
    buffer = ReplayBufferOld(
        buffer_size=40000,
        obs_shape=(221,),
        obs_dtype=np.float32,
        action_dim=1,
        action_dtype=np.int64,
        device=rank,
    )
    buffer.load_dataset(dataset)

    # create model
    model = BlackJackNet5b().to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    model = DDP(model, device_ids=[rank])

    model.train(True)
    iterations = 0  # training iterations
    optimizer = optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=1e-3
    )

    if config["load_ckpt"]:
        logger.info("Initialize model from checkpoint")
        ckpt_path = config["load_path"]
        if os.path.exists(ckpt_path):
            logger.info("Loading from checkpoint path found")
            ckpt = torch.load(ckpt_path, map_location=torch.device(rank))
            if config["load_from_sl"]:
                model.module.load_state_dict(ckpt)
                entropy_coeff *= 4e-4
                logger.info(f"Loading from SL checkpoint")
            else:
                model.load_state_dict(ckpt["state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                entropy_coeff = ckpt["entropy"]
                iterations = ckpt["iteration"]
                logger.info(
                    f"Loading from iteration {iterations}, effective entropy {entropy_coeff}"
                )
        else:
            logger.info("Checkpoint path not found, fall back to default init")

    sample_gen_rate = 0

    last_ckpt_time = None
    last_check_mutable_time = time.time()
    # if not receiving stop signal
    while not learner_status_list[1]:
        # while pause
        if learner_status_list[2]:
            time.sleep(0.5)
            continue

        # unpaused
        # if this is the first time, send model dict
        if rank == 0 and not first_model_sent:
            cpu_state_dict = {k: v.cpu() for k, v in model.module.state_dict().items()}
            tagged_cpu_state_dict = {
                "tag": model_tag_id,
                "state_dict": cpu_state_dict,
            }
            if model_pool_server is not None:
                model_pool_server.push(tagged_cpu_state_dict)
            else:
                raise ValueError("Model pool server not initialized")
            model_tag_id += 1
            logger.info("Sending Initial Model parameters")
            first_model_sent = True

        # receiving data and maintaining replay buffer
        sample_data_list = []
        try:
            while not replay_data_queue.empty():
                sample_data_patch = replay_data_queue.get(block=False)
                sample_data_list.append(sample_data_patch)
        except Exception:
            logger.info("Encountered replay buffer queue error, recovered")
            continue
        if len(sample_data_list) != 0:
            # rcvd_data_count, tag_dict = rb_svr.rcv_data_filtered(sample_data_list)
            rcvd_data_count = rb_svr.rcv_data(sample_data_list)
            sample_gen_rate += rcvd_data_count

        # if not receiving green status, sleep and
        if not learner_status_list[0]:
            time.sleep(0.5)
            continue

        # green light signal received, start training
        # sleep depending on sync/async, sleep time if async
        if sync:
            # wait for buffer refresh
            if tag_dict[model_tag_id - 1] < rb_size * 0.99:
                time.sleep(0.5)
                continue
        else:
            time.sleep(wait_time)

        # sample batch
        batch = rb_svr.sample(batch_size)

        obs = torch.tensor(batch["state"]).to(rank)
        states = obs
        actions = torch.tensor(batch["action"]).unsqueeze(-1).to(rank)
        advs = torch.tensor(batch["adv"])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs = advs.to(rank)
        targets = torch.tensor(batch["target"]).to(rank)
        match_info = batch["info"]

        # # Checking sample validity
        # mask_cpu = batch["state"]["action_mask"]
        # action_cpu = batch["action"]
        # for i in range(len(action_cpu)):
        #     if mask_cpu[i][action_cpu[i]] == 0:
        #         logger.info("WARNING: an action with mask = 0 detected")

        # calculate PPO loss
        model.train(True)  # Batch Norm training mode
        with torch.no_grad():
            states = obs
            old_logits = model(states)
        old_probs = F.softmax(old_logits, dim=1).gather(1, actions)
        old_log_probs = torch.log(old_probs + 1e-8).detach()
        frame_policy_loss = 0
        frame_value_loss = 0
        frame_entropy_loss = 0
        frame_logit_ratio = 0
        frame_gradient = 0
        frame_p_grad = 0
        frame_v_grad = 0
        frame_e_grad = 0
        for _ in range(config["epochs"]):
            demo_batch = buffer.sample(demo_batch_size)
            demo_states = demo_batch["observations"].to(rank)
            demo_actions = demo_batch["actions"].to(rank)
            sv_logits = model(demo_states)
            demo_logits = (
                F.one_hot(
                    demo_actions,
                    num_classes=2,
                )
                .squeeze_(dim=1)
                .type(torch.float)
            )
            sv_loss = torch.mean(F.cross_entropy(sv_logits, demo_logits))

            optimizer.zero_grad(set_to_none=True)
            states = obs
            logits = model(states)
            action_dist = torch.distributions.Categorical(logits=logits)
            probs = stable_softmax(logits, dim=1)
            probs = probs.gather(1, actions)
            log_probs = torch.log(probs + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs).squeeze(-1)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - config["clip"], 1 + config["clip"]) * advs
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            # value_loss = torch.mean(
            #     F.mse_loss(
            #         values.squeeze(-1),
            #         targets,
            #     )
            # )
            entropy_loss = torch.mean(action_dist.entropy())

            loss = (
                config["policy_coeff"] * policy_loss
                # + config["value_coeff"] * value_loss
                - entropy_coeff * entropy_loss
                + static_ratio * sv_loss
            )

            loss.backward()
            original_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.7, norm_type=2
            )
            optimizer.step()
            # detect nan in grad
            frame_policy_loss += policy_loss.item()
            frame_value_loss += 0  # value_loss.item()
            frame_entropy_loss += entropy_loss.item()
            frame_logit_ratio += torch.mean(ratio).item()
            frame_gradient += original_grad.item()
            # frame_p_grad += policy_grad.item()
            # frame_v_grad += value_grad.item()
            # frame_e_grad += entropy_grad.item()

        # lr_scheduler.step()
        # calculate performance indicators
        frame_policy_loss /= 1.0 * config["epochs"]
        frame_value_loss /= 1.0 * config["epochs"]
        frame_entropy_loss /= 1.0 * config["epochs"]
        frame_logit_ratio /= 1.0 * config["epochs"]
        frame_gradient /= 1.0 * config["epochs"]
        frame_p_grad /= 1.0 * config["epochs"]
        frame_v_grad /= 1.0 * config["epochs"]
        frame_e_grad /= 1.0 * config["epochs"]

        stat_queue.put(
            {
                "batch_size": batch_size,
                "policy_loss": frame_policy_loss * config["policy_coeff"],
                "value_loss": frame_value_loss * config["value_coeff"],
                "entropy_loss": frame_entropy_loss * entropy_coeff,
                "data_gen_rate": sample_gen_rate,
                "iteration": iterations,
                "logit_ratio": frame_logit_ratio,
                "original_grad": frame_gradient,
                "p_grad": frame_p_grad,
                "v_grad": frame_v_grad,
                "e_grad": frame_e_grad,
            }
        )
        # update entropy coeff
        entropy_coeff *= config["entropy_decay"]
        sample_gen_rate = 0

        t = time.time()
        # update mutables
        if t - last_check_mutable_time > config["log_interval"]:
            with open(os.path.join(mutable_param_path, "wait_time.json"), "r") as f:
                mutable_data = json.load(f)
                wait_time = mutable_data["wait"]
        # only rank = 0 performs chores
        if rank == 0:
            if sync or iterations % config["model_sync_iteration"] == 0:
                # push new model
                cpu_state_dict = {
                    k: v.cpu() for k, v in model.module.state_dict().items()
                }
                tagged_cpu_state_dict = {
                    "tag": model_tag_id,
                    "state_dict": cpu_state_dict,
                }
                if model_pool_server is not None:
                    model_pool_server.push(tagged_cpu_state_dict)
                else:
                    raise ValueError("Model pool server not initialized")
                # logger.info("Pushing Model with id: {}".format(model_tag_id))
                model_tag_id += 1

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
                        "entropy": entropy_coeff,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    path,
                )
                last_ckpt_time = t
        iterations += 1

    if rank == 0:
        # cleanup model pool server
        model_pool_server.cleanup()
    cleanup()
