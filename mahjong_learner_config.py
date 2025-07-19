learner_config = {
    "replay_buffer_size": 4100,
    "model_pool_size": 32,
    "model_pool_name": "model-pool2",
    "gamma": 0.99,
    "lambda": 0.98,
    "min_data_patch_count": 100,
    "effective_batch_size": 4096,
    "device_id": 0,
    "epochs": 5,
    "world_size": 1,
    "clip": 0.1,
    "lr": 1e-5,
    "policy_coeff": 1,
    "value_coeff": 1,
    "entropy_coeff": 1.5e-1,
    "entropy_decay": 0.99998,
    "log_interval": 60,
    "log_fan_interval": 3600,
    "ckpt_save_interval": 1800,
    "ckpt_save_path": "checkpoint/",
    "sync_ppo": False,
    "model_sync_iteration": 1,  # nullify if sync_ppo = True
    "mutable_param_path": "mutable_params/",
    "log_note": "Maze_MPPO_left_new",
    "load_ckpt": False,
    "load_from_sl": False,  # alternative loading and training scheme for sl ckpts
    # "load_path": "logs/learner_0302131657_V4_model_no_guide_SIL/checkpoint/model_50099.pt",
    "load_path": "Supervised_Learning/sl_ckpts/smth/46.pkl",
    # "load_path": "/home/llf/model_75416.pt",
}
guided_actor_count = 10
sampling_actor_count = 70
