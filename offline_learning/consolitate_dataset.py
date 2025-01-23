import os
import numpy as np


def consolidate_data(
    original_path,
    consolidate_path,
    split_count,
):

    file_list = os.listdir(original_path)
    file_list.sort()
    len_data = len(file_list)
    segment_length = len_data // split_count
    for i in range(split_count):
        obs_ = []
        next_obs_ = []
        action_ = []
        next_action_ = []
        reward_ = []
        done_ = []
        mask_ = []
        for file in file_list[i * segment_length : (i + 1) * segment_length]:
            path_to_file = os.path.join(original_path, file)
            npz = np.load(path_to_file)
            obs_.extend(npz["obs"])
            next_obs_.extend(npz["next_obs"])
            action_.extend(npz["action"])
            next_action_.extend(npz["next_actions"])
            reward_.extend(npz["rewards"])
            done_.extend(npz["is_done"])
            mask_.extend(npz["action_mask"])
        data = {
            "obs": obs_,
            "next_obs": next_obs_,
            "action": action_,
            "next_action": next_action_,
            "reward": reward_,
            "done": done_,
            "mask": mask_,
        }
        np.savez(os.path.join(consolidate_path, f"{i}.npz"), **data)


if __name__ == "__main__":
    original_dir = "../data/smth_offline_dataset"
    output_dir = "../data/smth_offline_dataset_consolidated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    split_count = 100
    consolidate_data(original_dir, output_dir, split_count)
