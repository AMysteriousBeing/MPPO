actor_config = {
    "model_pool_size": 32,
    "model_pool_name": "model-pool",
    "gamma": 1,
    "lambda": 0.98,
    "ckpt_save_path": "checkpoint/",
    "path_to_data": "NA",
    "path_to_eval_data": "NA",
    "core_config": [80, 10, 0],  # sampling core, true trajectory core, eval core
    "early_kill": False,
    "guide_trajectory": 0.1,
    "self_play": True,
    "init_hands": 31300,
    "augmentation": True,
    "data_iteration": 100,
    "log_dir": "NA",
}

