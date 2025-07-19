from torch.utils.data import Dataset
import numpy as np
import json
from multiprocessing import Pool
from bisect import bisect_right
import torch
import os


def workload(data_path, idx_list, start_idx):
    cache = {"obs": [], "mask": [], "act": []}
    for i in idx_list:
        # d = np.load("data/processed_data2/%d.npz" % (i + start_idx))
        d = np.load(os.path.join(data_path, "%d.npz" % (i + start_idx)))
        for k in d:
            cache[k].append(d[k])
    return cache


class MahjongGBDataset(Dataset):
    def __init__(self, data_path, meta_path, begin=0, end=1, num_workers=8):

        # with open("data/count2.json") as f:
        with open(meta_path) as f:
            self.match_samples = json.load(f)
        self.total_matches = len(self.match_samples)  # total number of matches
        self.total_samples = sum(
            self.match_samples
        )  # total number of trainable samples
        self.begin = int(begin * self.total_matches)  # start location by match
        self.end = int(end * self.total_matches)  # end location by match
        self.match_samples = self.match_samples[
            self.begin : self.end
        ]  # select by match
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        t = 0
        for i in range(self.matches):  # convert match samples to exclusive prefix sum
            a = self.match_samples[i]  # a: samples in one match
            self.match_samples[i] = t  # update i-th match to exclusive prefix sum
            t += a
        # At this point, self.match_samples contains exclusive prefix sum of sample matches (sum of total trainable rounds up to)

        self.cache = {"obs": [], "mask": [], "act": []}
        ret_list = []

        pool = Pool(num_workers)
        for i in range(num_workers):
            ret = pool.apply_async(
                workload,
                args=(
                    data_path,
                    range(
                        int(self.matches / num_workers * i),
                        int(self.matches / num_workers * (i + 1)),
                    ),
                    self.begin,
                ),
            )
            ret_list.append(ret)
        pool.close()
        pool.join()
        for ret in ret_list:
            d = ret.get()
            for k in d:
                self.cache[k].extend(d[k])

        for k in self.cache:
            self.cache[k] = np.concatenate(self.cache[k], axis=0)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):

        return (
            torch.tensor(self.cache["obs"][index]),
            torch.tensor(self.cache["mask"][index]),
            torch.tensor(self.cache["act"][index]),
        )


if __name__ == "__main__":
    from collections import defaultdict

    datapath = "../data/smth/sl_data_featurefull3"
    dataset = workload(datapath, range(0, 1), 0)
    obs = dataset["obs"]
    data_dict = defaultdict(int)
    for i in range(len(obs[0])):
        for j in obs[0][i]:
            data_dict[j] += 1
    print(data_dict)
    print(len(obs[0]), len(obs[0][0]))
