from offline_ddp_trainer import ddp_training
import torch.multiprocessing as mp
import datetime
import os

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    # Logging
    now = datetime.datetime.now()
    now = "learner_" + now.strftime("%m%d%H%M%S")
    if not os.path.exists("logs"):
        os.makedirs("logs")

    config = {
        "gamma": 0.99,
        "path_to_data": "../data/smth_offline_dataset_consolidated",
        "effective_batch_size": 4096,
        "world_size": 4,
        "clip": 0.1,
        "lr": 2e-5,
        "gamma": 0.98,
        "epoch": 200,
        "step_per_epoch": 1000,
        "log_interval": 60,
        "entropy": 1e-3,
        "ckpt_save_interval": 1800,
        "ckpt_save_path": "checkpoint/",
        "log_note": "offline_smth_cql",
        "seed": 0,
    }
    # Creating log folder
    config["log_dir"] = "{}_{}".format(now, config["log_note"])
    os.mkdir("logs/{}".format(config["log_dir"]))
    # ddp_training(0, config)
    mp.spawn(
        ddp_training,
        args=(config,),
        nprocs=config["world_size"],
        join=True,
    )
